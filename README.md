# Open AI Caribbean Challenge

**Team name: Tanacondas**

**members:**
- **Ákos Jakub**
- **Gábor Pelesz**
- **Zeynep Tasci**

## Final Report

Our work is summarized by the **final_report.pdf** file.

## Repository

- **Algorithms**: Part of our self-made algorithms which used by the model creation.
- **Extract images**: Module for extracting images from the Geospatial TIFF island images.
- **Milestones**: Milestones for the *BME TMIT Deep Learning based on Python* course
- **Predict**: The evaluation module which receives a .h5 file as input and creates the output for the test images in the submission format.
- **Train**: Our model scripts. For information check the **final_report.pdf** file.

## About our project

Our team was interested in image processing with deep learning since the beginning of the course. As a first thought we wanted to create a melanoma/skin cancer detecting neural network, but later we abandoned this idea.

Later on we found a fascinating competition on the website called Driven Data:
https://www.drivendata.org/competitions/58/disaster-response-roof-type/

The main topic of the task is also image processing; only with a more uncommon use of it. 

The regions of the **Caribbean**, **Central America**, the **South Pacific** and the **Himalayas** face a considerable number natural hazards every year, including earthquakes, hurricanes and floods. These can have a devastating effect due to building conditions in the region. In many cities, houses have been built without following sound construction standards. As such, many of these houses will likely be damaged or destroyed during the next natural disaster. What’s more, the majority of these houses are located in poor and often informal settlements that have grown over time to become very large and densely populated neighborhoods.

<br>

![title](https://th.thgim.com/news/international/809cbr/article29367424.ece/alternates/FREE_660/08IN-LT-TROPICALWEATHERBAHAMAS)

<br>
In order to retrofit relevant buildings in those very populated areas to bring them up to better construction standards, it is paramount that buildings that face higher risk of damage be identified quickly and accurately.

The traditional approach to identifying high-risk buildings is by foot: going door to door to visually assess building conditions, construction materials, roof types and other key factors that greatly influence how a building will fare during a natural disaster. This type of visual assessment is particularly time consuming and costly. A visual assessment by engineers typically takes weeks if not months and costs millions of dollars.

As a result, **WeRobotics** is teaming up with the **World Bank Global Program for Resilient Housing** to put drone imagery and AI image recognition to test the following hypothesis: the use of drones can help to quickly identify relevant buildings by creating risk categories that speed up the visual assessment process. Mapping a 10 km2 neighborhood with a drone can be done within a matter of days and at a cost of a few thousand dollars at most. The point here is not to replace onsite assessments of building conditions, but rather to narrow down the number of buildings that require onsite inspection before making a retrofit decision.

|  |  |
|---|---|
| <a href="https://werobotics.org/"><img src="https://blog.werobotics.org/wp-content/uploads/2017/06/Screenshot-2017-06-05-12.04.53.png" alt="WeRobotics" style="width: 380px;"/> | <a href="https://www.worldbank.org/en/topic/disasterriskmanagement/brief/global-program-for-resilient-housing"><img src="https://www.trzcacak.rs/myfile/detail/401-4016837_world-bank-group-logo.png" alt="WeRobotics" style="width: 400px;"/> |

The Global Program for Resilient Housing has assembled unique datasets with the goal of finding machine learning models that are able to most accurately map disaster risk from drone imagery. The objective is faster, cheaper prioritization of building inspections to help target resources for disaster preparation where they will have the most impact.

The goal of this challenge is to identify rooftop construction material. Roof construction material is one of the main building risks factors for earthquakes and hurricanes. Light material is more likely to fly off and leave people unprotected from strong wind, flying objects and rain during hurricanes; heavy material such as concrete is likely to collapse during an earthquake.
