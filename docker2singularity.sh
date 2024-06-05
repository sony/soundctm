DOCKER_IMAGE=soundctm

LOCALHOST=localhost
PN=5000

# pull docker registry
docker pull registry

# run docker registry at local
docker run  -d -p $PN:$PN --name tmp_registry registry

# set registry tag to local image
docker tag $DOCKER_IMAGE $LOCALHOST:$PN/$DOCKER_IMAGE

# push local image to local registry
docker push $LOCALHOST:$PN/$DOCKER_IMAGE

# cleanup if you need
docker rmi $LOCALHOST:$PN/$DOCKER_IMAGE
# docker rmi $DOCKER_IMAGE

# create singularity image with local registry
SINGULARITY_NOHTTPS=true NO_PROXY=localhost singularity pull docker://$LOCALHOST:$PN/$DOCKER_IMAGE

# creanup for local registry
docker kill tmp_registry
docker rm tmp_registry