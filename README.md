# REACH (Retrieval Engine for Articles and Common Household items)
## 1. Basic Overview 

Many times, everyday objects are misplaced within a person's home, causing them a lot of frustration and wasted time during urgent situations. In fact, the objects are located inside the house but we don’t the exact location. REACH Solves this problem.



## 2. Repository Contents

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

## 3. Dataset

The dataset for this REACH is generated through an application named RoboFlow, where we have annotated images, done all the pre-processing steps and divided the data into a training set, validation set and test set.


## 4. System Entry Point

Main script: src/main.py

This file consists our intended version of REACH where the camera opens up and detects the objects.

```sh
  streamlit run main.py
  ```

- **optional:**

(src/version2 contains the other version through which the user can get the detected objects output with a video input of objects.)


## 5. Video Demonstration

We explained in detail how the model and system function after deployment in the video demonstration. One can see where the images and feedbacks are getting saved.

## 6. Deployment Strategy

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




## 7. Project Documentation
AI System Project Proposal: documentation/AI System project proposal template

Project Report : documentation/Project report

## 8. Version Control and Team Collaboration

 

**Jugal Ganesh:** 

1.  Annotated 1000 images out of 2000 images in RoboFlow
2. Worked on main code version of object detection through webcam
3. Created several versions of pickle files.
4. Finalized deployment strategies for streamlit


**Naveen:**

1. Annotated 1000 images out of 2000 images in RoboFlow
2. Worked on input video version of REACH which lead to a breakthough in finalsing the models.
3. Worked on Feedback mechanism.
4. Tested mutiple versions of code and finalised the correct implementation




## 9. About Grafana
We haven't used prometheus and grafana instead we have used a tool named wandb.ai to measure our system metrics this tool gave us all the system metrics incuding CPU, GPU, memory utilization, CPU threads and temperatures.  

## 10.  If Not Applicable

You might face issues deploying the main version of REACH which uses a webcam. Do remember that docker can't detect local webcam so run it through streamlit and use version2 for docker implementation.


## 11. Contributing

We welcome pull requests. Before making any significant modifications, please start an issue to discuss your preferences.

## 12.Acknowledgements

We would like to thank our professor **Dr. Ramirez** for making this possible in limited time. We got help whenever required and able to finish up all the tasks in time.


## **Thank You**
