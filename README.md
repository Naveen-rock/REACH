# REACH (Retrieval Engine for Articles and Common Household items)
## Project Overview

**REACH** is an AI-powered system that helps users locate misplaced household items such as keys, wallets, and chargers in real-time. By leveraging advanced machine learning models and object tracking algorithms, REACH provides an easy-to-use interface that allows users to find everyday items effortlessly, reducing stress and improving daily efficiency.


## Objective

- **Problem:** Everyday items are often misplaced, leading to frustration and wasted time. Muscle memory and rushed routines can also cause users to overlook items, even when they are in plain sight.
- **Solution:** REACH detects and tracks misplaced objects in a user’s home, reports their approximate location, and displays this information through a simple interface.

### Expected Outcomes

1. **Object Location:** Provides snapshots and approximate locations of misplaced items.
2. **User-Friendly Interface:** A web-based platform with clear navigation and visual representations.
3. **Enhanced Convenience:** Integrates seamlessly into daily routines, allowing users to find items effortlessly.


## Technologies Used

- **Machine Learning:** YOLOv8 for object detection.
- **Tracking:** DeepSORT for real-time multi-object tracking.
- **Frameworks:** Streamlit for the user interface.
- **Deployment:** Docker for containerization.
- **Monitoring:** Wandb.ai for system performance metrics.
- **Dataset:** Annotated and preprocessed via RoboFlow.


## How It Works

1. The system uses a live camera feed or video input to detect objects.
2. **YOLOv8** and **DeepSORT** algorithms identify and track the objects in real-time.
3. The user interface displays results (snapshots and object locations) using **Streamlit**.
4. System performance and user feedback are logged via **Wandb.ai**.


## Repository Contents

-**src:**
 
This contains the core files for this project, i.e., ML code, Streamlit initialising, requirements for required libraries, trained pickle files.

-**deployment:**

Contains the required Dockerfile and docker-compose file as these are used in the deployment phase.

-**monitoring:**

This has all the images of the Wandb dashboard, which measured our system analytics while running the model.

-**documentaion:**

This section can help the people who visit this repository understand what this project is about and a detailed report to get deeper insights.

-**videos:**

A run-through video for better visualization and understanding of the project.

## Dataset

The dataset for this REACH is generated through an application named RoboFlow, where we have annotated images, done all the pre-processing steps and divided the data into a training set, validation set and test set.


## System Entry Point

Main script: src/main.py

This file consists our intended version of REACH where the camera opens up and detects the objects.

```sh
  streamlit run main.py
  ```

- **optional:**

(src/version2 contains the other version through which the user can get the detected objects output with a video input of objects.)


## Video Demonstration

We explained in detail how the model and system function after deployment in the video demonstration. One can see where the images and feedbacks are getting saved.

## Deployment Strategy

- Initially, We deployed our model to *streamlit* to run it locally and then we pushed that to *docker* platform and created images and containers to run them. We are running it on port 8501.

- However, the main version doesn't work on docker as docker can't access the local hardware(Camera). To use the main version run through **streamlit**.

- For Version 2, you can use the same model as of main.py
- docker-compose file is optional 

To create the docker image and run the container below is the code.

## Build the Docker image

```sh
  docker build -t reach .
  ```
## Run the container

```sh
  docker run -p 8501:8501 reach
  ```




## Project Documentation
AI System Project Proposal: documentation/AI System project proposal template

Project Report : documentation/Project report

## Version Control and Team Collaboration
 
**Naveen:**

1. Annotated 1000 images out of 2000 images in RoboFlow
2. Worked on input video version of REACH which lead to a breakthough in finalsing the models.
3. Worked on Feedback mechanism.
4. Tested mutiple versions of code and finalised the correct implementation

**Jugal Ganesh:** 

1. Annotated 1000 images out of 2000 images in RoboFlow
2. Worked on main code version of object detection through webcam
3. Created several versions of pickle files.
4. Finalized deployment strategies for streamlit

## About Grafana
We haven't used prometheus and grafana instead we have used a tool named wandb.ai to measure our system metrics this tool gave us all the system metrics incuding CPU, GPU, memory utilization, CPU threads and temperatures.  

##  If Not Applicable

You might face issues deploying the main version of REACH which uses a webcam. Do remember that docker can't detect local webcam so run it through streamlit and use version2 for docker implementation.


##  Contributing

We welcome pull requests. Before making any significant modifications, please start an issue to discuss your preferences.

## Acknowledgements

We would like to thank our professor **Dr. Ramirez** for making this possible in limited time. We got help whenever required and able to finish up all the tasks in time.


## **Thank You**
