"""Exercise the live IPC handlers with a held-clock CPU collector stand-in."""
from pathlib import Path
from types import SimpleNamespace
import math,tempfile,unittest
import numpy as np
from native_policy_bridge import capture_observation,apply_action
from wrist_retention import WristHistory

FEATURES=('shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos')

class HeldCollector:
    def __init__(self,rtf):
        self.rtf,self.steps,self.image_reads=rtf,0,[]
        self.session=self.sensors=self
        self.set_current([0.]*5+[1.])
    def set_current(self,q):
        stamp=1_000_000_000+round(self.steps*1e9/30)
        self.current=dict(sim_ns=stamp,epoch=9,state=list(q),fake_wall_s=self.steps/30/self.rtf)
        self.images={cam:np.full((480,640,3),self.steps,np.uint8) for cam in ('top','wrist')}
        self.entry={cam:(cam,None,dict(stream=cam,stamp_ns=stamp,native_tick=dict(epoch=9,sim_ns=stamp))) for cam in self.images}
    def image(self,key):self.image_reads.append(key);return self.images[key]
    def step(self,command):
        self.steps+=1;self.set_current(command)
        return dict(ack=dict(applied_command=list(command))),self.current

class NativeBridgeTests(unittest.TestCase):
    def rollout(self,rtf,retain):
        transport=HeldCollector(rtf);history=WristHistory(True,max_actions=45) if retain else None
        commands=[];snaps=[transport.current];captured=[];first={};scorer=SimpleNamespace(update=lambda state:None)
        for i in range(45):
            if i%10==0:
                obs,meta=capture_observation(transport,FEATURES,transport.current,first,history,len(commands));captured.append(meta)
                self.assertEqual(obs['top'].shape,(480,640,3));self.assertAlmostEqual(meta['state_degrees'][5],math.degrees(transport.current['state'][5]))
            action={name:float(i) if k<5 else math.degrees(1. if i<29 else .9) for k,name in enumerate(FEATURES)}
            reply=apply_action(transport,FEATURES,np.asarray([[-math.pi,math.pi]]*6),[0.]*6,False,action,commands,snaps,scorer,history)
            self.assertAlmostEqual(reply['performed']['shoulder_pan.pos'],float(i))
        self.assertEqual(transport.steps,45);self.assertEqual(len(commands),45);self.assertEqual(len(snaps),46)
        self.assertTrue(np.all(first['wrist']==0))
        return transport,history,commands,snaps,captured
    def test_telemetry_changes_no_native_actions_or_timing_at_either_rtf(self):
        reference=None
        for rtf in (.3,1.):
            plain,_,commands,snaps,captures=self.rollout(rtf,False)
            retained,h,other,other_snaps,other_captures=self.rollout(rtf,True)
            self.assertEqual(commands,other);self.assertEqual(captures,other_captures)
            self.assertEqual([s['sim_ns'] for s in snaps],[s['sim_ns'] for s in other_snaps])
            self.assertEqual(plain.image_reads.count('wrist'),5);self.assertEqual(retained.image_reads.count('wrist'),50)
            if reference is None:reference=commands
            else:self.assertEqual(reference,commands)
            with tempfile.TemporaryDirectory() as t:
                proof=h.finish(Path(t)/'wrist');self.assertEqual(proof['closures_recorded'],1);self.assertEqual(proof['frames_saved'],31)
                self.assertEqual(proof['verified_native_ticks'],46);self.assertEqual(proof['executed_actions'],45)
                self.assertEqual(proof['last_sim_ns']-proof['first_sim_ns'],1_500_000_000)
    def test_warm_hold_and_nonfinite_refusal(self):
        t=HeldCollector(1.);scorer=SimpleNamespace(update=lambda state:None);commands=[];snaps=[]
        action=dict.fromkeys(FEATURES,100.);hold=[0.]*5+[1.]
        reply=apply_action(t,FEATURES,np.asarray([[-math.pi,math.pi]]*6),hold,True,action,commands,snaps,scorer,None)
        self.assertEqual(t.current['state'],hold);self.assertTrue(commands[0]['warmup_hold'])
        action[FEATURES[0]]=math.nan
        with self.assertRaises(ValueError):apply_action(t,FEATURES,np.asarray([[-math.pi,math.pi]]*6),hold,False,action,commands,snaps,scorer,None)
        self.assertEqual(t.steps,1)

if __name__=='__main__':unittest.main()
