"""Bounded, directory-pinned copies into our own dual backup projection."""
import fcntl,hashlib,inspect,json,os,stat,tempfile,uuid
from contextlib import contextmanager
from pathlib import Path

BASE=Path('/mnt/extreme4tb/aaugus11/exploring-vlas')
PROJECTION=BASE/'datasets/counterfactual-production-v1'


def require_capacity(free_bytes,incoming_bytes,file_count):
    if any(type(v) is not int or v<0 for v in (free_bytes,incoming_bytes,file_count)):
        raise ValueError('Invalid backup capacity measurement')
    floor=100*1024**3
    overhead=64*1024**2+file_count*16*1024
    required=floor+incoming_bytes+overhead
    if free_bytes<required:
        raise RuntimeError('Dual /mnt/extreme4tb would cross 100 GiB floor: free_GiB='+format(free_bytes/1024**3,'.3f')+', free_bytes='+str(free_bytes)+
            ', incoming='+str(incoming_bytes)+', overhead='+str(overhead)+', required='+str(required))
    return dict(passed=True,free_bytes=free_bytes,incoming_bytes=incoming_bytes,file_count=file_count,
        floor_bytes=floor,overhead_bytes=overhead,required_free_bytes=required,
        projected_free_bytes=free_bytes-incoming_bytes-overhead)


def check_chain(chain):
    for index,(fd,old) in enumerate(chain):
        now=os.fstat(fd) if index==0 else os.stat(Path(old['path']).name,dir_fd=chain[index-1][0],follow_symlinks=False)
        if not stat.S_ISDIR(now.st_mode) or (now.st_dev,now.st_ino)!=(old['device'],old['inode']):
            raise ValueError('Backup directory identity changed: '+old['path'])


@contextmanager
def directory_chain(path,create=False,expected=()):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('Invalid backup directory')
    chain=[];known={row['path']:row for row in expected}
    try:
        parent=None
        for index,part in enumerate(path.parts):
            current=Path(*path.parts[:index+1]);check_chain(chain)
            try:fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            except FileNotFoundError:
                if not create:break
                os.mkdir(part,0o700,dir_fd=parent);os.fsync(parent)
                fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            except OSError as exc:raise ValueError('Backup ancestor is not a real directory: '+str(current)) from exc
            info=os.fstat(fd);row=dict(path=str(current),device=info.st_dev,inode=info.st_ino)
            chain.append((fd,row));parent=fd
            if str(current) in known and row!=known[str(current)]:raise ValueError('Backup directory identity changed: '+str(current))
        yield chain
    finally:
        for fd,_ in reversed(chain):os.close(fd)


def validate_admission(admission,destination,specs):
    if (not Path(destination).is_relative_to(PROJECTION) or '..' in Path(destination).parts or
        admission.get('destination')!=str(destination) or admission.get('filesystem_root')!=str(BASE)):
        raise RuntimeError('Dual backup capacity admission was not bound to this copy')
    incoming=0
    for name,row in specs.items():
        path=Path(name)
        if (not isinstance(name,str) or not name or path.is_absolute() or '..' in path.parts or
            str(path)!=name or name=='.' or '\n' in name or '\0' in name or
            type(row.get('size')) is not int or row['size']<0 or
            not isinstance(row.get('sha256'),str) or len(row['sha256'])!=64 or
            any(c not in '0123456789abcdef' for c in row['sha256'])):raise ValueError('Invalid manifested backup file: '+str(name))
        incoming+=row['size']
    expected=require_capacity(admission['free_bytes'],incoming,len(specs))
    if any(admission.get(k)!=v for k,v in expected.items()):raise RuntimeError('Dual backup capacity admission was not bound to this copy')
    identities=admission.get('directory_identities')
    ancestors=list(reversed((Path(destination),*Path(destination).parents)))
    if (not isinstance(identities,list) or not identities or str(BASE) not in [r.get('path') for r in identities] or
        len(identities)>len(ancestors) or any(r.get('path')!=str(ancestors[i]) or
            any(type(r.get(k)) is not int or r[k]<0 for k in ('device','inode')) for i,r in enumerate(identities))):
        raise RuntimeError('Dual backup directory admission was not bound to this copy')
    return incoming


def probe_code(destination,incoming_bytes,file_count):
    return common_code()+inspect.getsource(require_capacity)+f'''
destination=Path({str(destination)!r})
if '..' in destination.parts or not destination.is_relative_to(PROJECTION):raise ValueError('Backup destination is outside our projection')
with directory_chain(destination) as chain:
 check_chain(chain)
 base=next(fd for fd,row in chain if row['path']==str(BASE))
 space=os.fstatvfs(base)
 result=require_capacity(space.f_bavail*space.f_frsize,{incoming_bytes!r},{file_count!r})
 result.update(destination=str(destination),filesystem_root=str(BASE),directory_identities=[row for fd,row in chain],checked_wall_ns=__import__('time').time_ns())
 print(json.dumps(result))
'''


def common_code():
    return ('import hashlib,json,os,stat,sys,uuid\nfrom pathlib import Path\nfrom contextlib import contextmanager\n'+
        'BASE=Path('+repr(str(BASE))+');PROJECTION=Path('+repr(str(PROJECTION))+')\n'+
        inspect.getsource(check_chain)+inspect.getsource(directory_chain))


@contextmanager
def sealed_payload(source,specs,admission,directory):
    """Only fixed sizes and hashes enter this private, unlinked payload file."""
    with tempfile.TemporaryFile(dir=directory) as wire:
        header=json.dumps(dict(version=1,destination=admission['destination'],files=specs,admission=admission)).encode()+b'\n'
        if len(header)>8*1024**2:raise ValueError('Backup manifest header is too large')
        wire.write(header)
        for name,spec in sorted(specs.items()):
            path=Path(name)
            with directory_chain(Path(source)/path.parent) as chain:
                if chain[-1][1]['path']!=str(Path(source)/path.parent):raise ValueError('Backup source directory disappeared')
                check_chain(chain);fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=chain[-1][0])
                with os.fdopen(fd,'rb') as stream:
                    info=os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size!=spec['size']:raise ValueError('Backup source size/type changed: '+name)
                    digest=hashlib.sha256();remaining=spec['size']
                    while remaining:
                        block=stream.read(min(1024**2,remaining))
                        if not block:raise ValueError('Backup source shortened: '+name)
                        wire.write(block);digest.update(block);remaining-=len(block)
                    if stream.read(1) or digest.hexdigest()!=spec['sha256'] or os.fstat(stream.fileno()).st_mtime_ns!=info.st_mtime_ns:
                        raise ValueError('Backup source changed: '+name)
                check_chain(chain)
        wire.flush();wire.seek(0);yield wire


def receive_backup(destination,stream):
    header=stream.readline(8*1024**2+1)
    if len(header)>8*1024**2 or not header.endswith(b'\n'):raise ValueError('Invalid bounded backup header')
    request=json.loads(header);specs=request['files'];admission=request['admission']
    if request['version']!=1 or request['destination']!=str(destination):raise ValueError('Backup transfer destination differs')
    incoming=validate_admission(admission,destination,specs);identities={r['path']:r for r in admission['directory_identities']}
    with directory_chain(BASE,expected=identities.values()) as base_chain:
        if base_chain[-1][1]['path']!=str(BASE):raise ValueError('Backup filesystem root is missing')
        base_fd=base_chain[-1][0];space=os.fstatvfs(base_fd)
        capacity=require_capacity(space.f_bavail*space.f_frsize,incoming,len(specs))
        with directory_chain(destination,create=True,expected=identities.values()) as pinned:
            identities.update((r['path'],r) for _,r in pinned)
            for name,spec in sorted(specs.items()):
                check_chain(pinned);path=Path(name)
                with directory_chain(destination/path.parent,create=True,expected=identities.values()) as chain:
                    identities.update((r['path'],r) for _,r in chain);parent=chain[-1][0]
                    temp='.backup-'+uuid.uuid4().hex
                    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
                    with os.fdopen(fd,'wb') as output:
                        remaining=spec['size'];digest=hashlib.sha256()
                        while remaining:
                            check_chain(pinned);check_chain(chain)
                            block=stream.read(min(1024**2,remaining))
                            if not block:raise ValueError('Backup transfer shortened: '+name)
                            output.write(block);digest.update(block);remaining-=len(block)
                        output.flush();os.fsync(output.fileno())
                    if digest.hexdigest()!=spec['sha256']:raise ValueError('Backup transfer SHA differs: '+name)
                    check_chain(pinned);check_chain(chain)
                    try:old=os.stat(path.name,dir_fd=parent,follow_symlinks=False)
                    except FileNotFoundError:old=None
                    if old is not None and not stat.S_ISREG(old.st_mode):raise ValueError('Backup leaf is not regular: '+name)
                    os.replace(temp,path.name,src_dir_fd=parent,dst_dir_fd=parent);os.fsync(parent)
            if stream.read(1):raise ValueError('Backup transfer has unadmitted bytes')
            for name,spec in sorted(specs.items()):
                path=Path(name);check_chain(pinned)
                with directory_chain(destination/path.parent,expected=identities.values()) as chain:
                    if chain[-1][1]['path']!=str(destination/path.parent):raise ValueError('Backup directory disappeared')
                    fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=chain[-1][0])
                    with os.fdopen(fd,'rb') as source:
                        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):raise ValueError('Backup verification leaf is not regular')
                        digest=hashlib.sha256()
                        while block:=source.read(1024**2):digest.update(block)
                        if source.tell()!=spec['size'] or digest.hexdigest()!=spec['sha256']:raise ValueError('Backup verification SHA differs: '+name)
                    check_chain(chain)
            check_chain(base_chain);check_chain(pinned);space=os.fstatvfs(base_fd)
            if space.f_bavail*space.f_frsize<capacity['floor_bytes']:raise RuntimeError('Dual backup filesystem fell below 100 GiB')
            return dict(passed=True,destination=str(destination),files_sha256={n:r['sha256'] for n,r in specs.items()},
                transferred_bytes=incoming,capacity_admission=admission,transfer_capacity=capacity,
                directory_identities=list(identities.values()),destination_identity=pinned[-1][1])


def receiver_code(destination):
    return (common_code()+inspect.getsource(require_capacity)+inspect.getsource(validate_admission)+inspect.getsource(receive_backup)+
        '\nprint(json.dumps(receive_backup(Path('+repr(str(destination))+'),sys.stdin.buffer)))\n')


@contextmanager
def serialized_copy(path):
    """Serialize our four copies through admission, transfer and verification."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise RuntimeError('Backup lock is not regular')
        fcntl.flock(fd,fcntl.LOCK_EX)
        yield
    finally:os.close(fd)


MAX_CHECK_SCRIPT_BYTES=8*1024**2


def checked_script_payload(code):
    """Frame a read-only verification program without putting it in argv."""
    body=code.encode('utf-8')
    if not body or len(body)>MAX_CHECK_SCRIPT_BYTES:raise ValueError('Backup check script exceeds bounded capacity')
    header=dict(version=1,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    return json.dumps(header,separators=(',',':')).encode()+b'\n'+body


def receive_check_script(stream):
    header=stream.readline(513)
    if len(header)>512 or not header.endswith(b'\n'):raise ValueError('Invalid backup check header')
    value=json.loads(header)
    if (not isinstance(value,dict) or set(value)!={'version','bytes','sha256'} or type(value['version']) is not int or value['version']!=1 or
        type(value['bytes']) is not int or not 0<value['bytes']<=MAX_CHECK_SCRIPT_BYTES or
        not isinstance(value['sha256'],str) or len(value['sha256'])!=64 or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('Invalid backup check declaration')
    body=bytearray();remaining=value['bytes']
    while remaining:
        block=stream.read(min(1024**2,remaining))
        if not block:raise ValueError('Backup check input truncated')
        body.extend(block);remaining-=len(block)
    if stream.read(1):raise ValueError('Backup check input has extra bytes')
    if hashlib.sha256(body).hexdigest()!=value['sha256']:raise ValueError('Backup check input SHA differs')
    return body.decode('utf-8')


def check_script_receiver_code():
    # This constant-sized bootstrap validates all input before executing the
    # caller's unchanged read-only verification program.
    return ('import hashlib,json,sys\nMAX_CHECK_SCRIPT_BYTES='+str(MAX_CHECK_SCRIPT_BYTES)+'\n'+
        inspect.getsource(receive_check_script)+'\nexec(compile(receive_check_script(sys.stdin.buffer),"<verified-backup-check>","exec"),{})\n')
