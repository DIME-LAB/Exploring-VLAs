"""Opt-in collector-held policy timing; retain production RPC and processors."""
def install():
    from lerobot.async_inference import policy_server as production
    import os
    if os.environ.get('SOARM_EVAL_NATIVE_HELD')!='1':raise RuntimeError('Native held-tick server requires explicit opt-in')
    cap=os.environ.get('SOARM_EVAL_TORCH_MEMORY_GIB')
    if cap:
        import torch
        limit=float(cap)*1024**3
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('GPU memory cap requires one visible CUDA device')
        torch.cuda.set_per_process_memory_fraction(limit/torch.cuda.get_device_properties(0).total_memory,device=0)
    native_factory=production.get_policy_class
    def factory(name):
        cls=native_factory(name)
        class Loader:
            @staticmethod
            def from_pretrained(path):
                from lerobot.configs.policies import PreTrainedConfig
                config=PreTrainedConfig.from_pretrained(path);config.compile_model=False
                return cls.from_pretrained(path,config=config)
        return Loader
    production.get_policy_class=factory
    native=production.PolicyServer
    class HeldPolicyServer(native):
        def _obs_sanity_checks(self,obs,previous_obs):
            with self._predicted_timesteps_lock:return obs.get_timestep() not in self._predicted_timesteps
        def _get_action_chunk(self,observation):
            raw=self.last_processed_obs.get_observation();context=raw['_eval_native_held']
            if context.get('mode')!='collect' or context.get('held') is not True or context.get('version')!=1:
                raise RuntimeError('Missing held native observation identity')
            step=self.last_processed_obs.get_timestep();previous=self._rtc_previous_chunk;offset=step-self._rtc_chunk_start
            tail=previous[:,offset:].clone() if previous is not None and 0<=offset<previous.shape[1] else None
            rtc=getattr(self.policy.config,'rtc_config',None);enabled=rtc is not None and rtc.enabled
            # The collector cannot advance while this synchronous request waits.
            # The client measures the native stamp again at arrival and refuses
            # the response if physics advanced: measured inference delay is0.
            kwargs=dict(prev_chunk_left_over=tail,inference_delay=0) if enabled else {}
            self.native_rtc=dict(enabled=enabled,step=step,prefix_steps=tail.shape[1] if enabled and tail is not None else 0,inference_delay=0,capture_sim_ns=context['sim_ns'],epoch=context['epoch'],execution_horizon=rtc.execution_horizon if enabled else None,max_guidance_weight=rtc.max_guidance_weight if enabled else None)
            if context.get('flow_noise') is not None:
                from flow_noise import prediction_noise
                state=observation['observation.state']
                noise,proof=prediction_noise(context['flow_noise'],step,state.shape[0],self.policy.config.chunk_size,self.policy.config.max_action_dim,state.device)
                kwargs['noise']=noise;self.native_rtc['flow_noise']=proof
            chunk=self.policy.predict_action_chunk(observation,**kwargs)
            if chunk.ndim!=3:chunk=chunk.unsqueeze(0)
            chunk=chunk[:,:self.actions_per_chunk,:];self._rtc_previous_chunk=chunk.detach().clone();self._rtc_chunk_start=step
            return chunk
        def _predict_action_chunk(self,observation_t):
            result=super()._predict_action_chunk(observation_t)
            if result:result[0].native_rtc=dict(self.native_rtc)
            return result
    production.PolicyServer=HeldPolicyServer
if __name__=='__main__':
    install()
    from lerobot.async_inference.policy_server import serve
    serve()
