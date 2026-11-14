# Shared by indexed launchers and the smoke runner. Inherited descriptors must
# name this exact lock file and hold its exclusive flock before any startup.
claim_collection_instance() {
    local root="$1" new_claim
    exec {new_claim}>> "$root/instance.lock"
    if [ -n "${SOARM_INSTANCE_CLAIM_FD:-}" ]; then
        python3 - "$SOARM_INSTANCE_CLAIM_FD" "$root/instance.lock" <<'PYCLAIM'
import fcntl,os,sys
fd=int(sys.argv[1]);opened=os.fstat(fd);expected=os.stat(sys.argv[2],follow_symlinks=False)
assert (opened.st_dev,opened.st_ino)==(expected.st_dev,expected.st_ino),'Inherited instance claim names another file'
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
PYCLAIM
        if [ "$?" -ne 0 ]; then echo 'Invalid or occupied instance claim' >&2; exit 1; fi
        exec {new_claim}>&-
    else
        flock -n "$new_claim" || { echo 'Private instance is already claimed' >&2; exit 1; }
        export SOARM_INSTANCE_CLAIM_FD="$new_claim"
    fi
}
