"""Content identity for once-only sensor joins; approximate stamps never dedupe content."""
import hashlib,json


class SensorAmbiguity(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('Ambiguous sensor frame candidates: '+json.dumps(evidence,sort_keys=True))


def identity(key,message,stamp):
    if key in ('wrist','top'):
        layout={name:getattr(message,name,None) for name in ('encoding','width','height','step','is_bigendian')}
        layout['frame_id']=getattr(message.header,'frame_id','')
        content=hashlib.sha256(json.dumps(layout,sort_keys=True).encode())
        content.update(bytes(getattr(message,'data',b'')))
    else:
        payload=message if key=='tick' else {name:list(getattr(message,name,[])) for name in ('name','position','velocity','effort')}
        content=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=True).encode());layout=None
    return dict(stream=key,stamp_ns=stamp,content_sha256=content.hexdigest(),layout=layout)


def exact_duplicate(left,right):
    return left['stamp_ns']==right['stamp_ns'] and left['content_sha256']==right['content_sha256']
