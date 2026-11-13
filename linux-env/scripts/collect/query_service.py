"""Collector query failures stay explicit; GUI keeps its raw service semantics."""
class QueryFuture:
    def __init__(self,future,label,fail):self.future,self.label,self.fail=future,label,fail
    def done(self):return self.future.done()
    def result(self):
        try:
            if not self.future.done():raise TimeoutError(self.label+' query did not complete')
            result=self.future.result()
            if result is None:raise RuntimeError(self.label+' query returned no response')
            return result
        except Exception as exc:
            self.fail(exc)
            raise
    def __getattr__(self,name):return getattr(self.future,name)


class StrictQueryClient:
    def __init__(self,client,label,on_failure=lambda exc:None):
        self.client,self.label,self.fail=client,label,on_failure
    def service_is_ready(self):
        try:
            if not self.client.service_is_ready():raise RuntimeError(self.label+' service is unavailable')
            return True
        except Exception as exc:
            self.fail(exc)
            raise
    def call_async(self,request):
        try:return QueryFuture(self.client.call_async(request),self.label,self.fail)
        except Exception as exc:
            self.fail(exc)
            raise
    def __getattr__(self,name):return getattr(self.client,name)
