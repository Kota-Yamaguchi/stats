
FROM ubuntu

Run apt-get update && apt-get install -y python3 python3-pip
Run pip3 install numpy matplotlib pandas scipy

COPY * /~/ 
