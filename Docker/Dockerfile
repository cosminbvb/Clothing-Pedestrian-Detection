# Starting image
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update
RUN apt install -y python3 python3-pip
RUN apt-get install -y libgl1-mesa-dev

# Install YOLOv5 requirements
RUN pip3 install --upgrade pip
RUN pip3 install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

RUN pip3 install flask

# Copy to image
COPY ./startApp.sh startApp.sh

# Execute script
ENTRYPOINT [ "./startApp.sh" ]

