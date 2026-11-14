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


def publisher_identity(info):
    def get(key):return info.get(key) if isinstance(info,dict) else getattr(info,key,None)
    raw=get('publisher_gid');implementation=get('implementation_identifier')
    try:gid=bytes(raw)
    except (TypeError,ValueError):gid=b''
    if len(gid)!=24 or not any(gid) or not isinstance(implementation,str) or not implementation:
        raise SensorAmbiguity(dict(reason='packet publisher identity unavailable',publisher_gid=gid.hex(),implementation=implementation))
    return dict(scope='receiving subscription/context-local RMW identity',gid=gid.hex(),implementation=implementation,
        publication_sequence_number=(None if get('publication_sequence_number')==2**64-1 else get('publication_sequence_number')),source_timestamp=get('source_timestamp'),
        received_timestamp=(get('received_timestamp') or None),transport_take_decode_ms=get('transport_take_decode_ms'))


def publisher_key(publisher):return publisher['implementation']+':'+publisher['gid']


CAMERA_SELECTION_POLICY=dict(version=2,selection='first taken packet per exact stamp and publisher GID',
    rerender='same GID + stamp + layout: retain first pixels; count later renders',
    ambiguity='different GID, changed layout or different nearby stamp is terminal',
    boundary='discard stamps at/below cancellation floor plus 1 us producer tolerance')
