image = "jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-nemo:25.11.01-alps6"
mounts = ["/capstor", "/iopsstor", "/users"]
workdir = "/"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
[annotations]
com.hooks.cxi.enabled = "false"
