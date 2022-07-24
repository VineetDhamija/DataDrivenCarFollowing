<p align="center"> 
  <img src="images/Logo.png" alt="HAR Logo" >
</p>
<h1 align="center"> Data Driven Car Following Model </h1>
<h3 align="center"> A Comparison and Application for Random Forest, CNN and KNN Car Following models to predict Acceleration and calculate Subject Vehicle Trajectory</h3>  

</br>

<!---
<p align="center"> 
  <img src="images/Signal.gif" alt="Sample signal"> width="70%" height="70%">
</p>
-->

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project"> ➤ About The Project</a>
      <ul>
        <li><a href="#Problem-statement">Problem Statement</a></li>
        <li><a href="#Proposed-Solution">Proposed Solution</a></li>
      </ul>
    </li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#roadmap"> ➤ Roadmap</a></li>
    <li>
      <a href="#preprocessing"> ➤ Preprocessing</a>
      <ul>
        <li><a href="#Data-Cleanup">Data Cleanup</a></li>
        <li><a href="#Data-Transormation">Data Transormation</a></li>
      </ul>
    </li>
    <!-- Start From here. copleted till Preprocessing-->
    <li><a href="#results-and-discussion"> ➤ Results and Discussion</a></li>
    <li><a href="#references"> ➤ References</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
    <li><a href="#contributors"> ➤ Steps to Re-trace</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
  This project aims to understand and explore trajectories of various Car Following Models that predict Acceleration. The raw NGSIM data will be preprocessed to cleanup bad data and filter only valid pairs that would help train the model on I-80 and US-101 Highways. 
  The goal is to compare and evaluate the performance of different Models (Random Forest, k Nearest Neighbors, and CNN) and discuss the trajectories along with their R^2 and RMSE.
</p>
<!-- PRE-PROCESSED DATA -->
<h2 id="Problem-statement"> :diamond_shape_with_a_dot_inside: Problem Statement</h2>

<p align="justify"> 

* Very few data driven models predict acceleration accurately. 
*  Even with good RMSE scores, the trajectories do not match or aren’t displayed.
* The papers either falsely claim good acceleration predictions or completely avoid mentioning it.

</p>
<!-- PRE-PROCESSED DATA -->
<h2 id="Proposed-Solution"> :diamond_shape_with_a_dot_inside: Proposed Solution</h2>

<p align="justify"> 

 - Create and compare Data Driven car following model using NGSIM dataset.
 - Display and compare trajectories along with R2 and RMSE.

</p>

<p align="center">
  <img src="images/WISDM Activities.png" alt="Table1: 18 Activities" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20for-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>
[![made-with-VSCode](https://img.shields.io/badge/Made%20with-VSCode-1f425f.svg)](https://code.visualstudio.com/) <br>

<!--This project is written in Python programming language. <br>-->
The following open source packages are used in this project:
* Numpy
* Pandas
* TensorFlow
* Keras
* Matplotlib
* Sklearn
* pathlib
* random
* seaborn
* pickle


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    datadrivencarfollowing-v1
    .
    │
    ├── APP
    │   ├── Demo.py
    │   ├── Prediction_set_Predited_data_0.1.csv
    │   ├── Prediction_set_Predited_data_0.2.csv
    │   ├── Prediction_set_Predited_data_0.3.csv
    │   ├── Prediction_set_Predited_data_0.5.csv
    │   ├── Prediction_set_Predited_data_1.csv
    │   ├── Prediction_set_Predited_data_2.csv
    │   └── Prediction_set_Predited_data_4.csv
    ├── data
    │   ├── all Datasets required for code or created by code.
    │   └── Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv ==> Required Input NGSIM file name to recreate
    ├── docs
    │   └── Data Dictionary.xlsx
    ├── results
    │   ├── Car Following Model.pptx
    │   ├── Data-Cleanup_Transformation_Test_Train_Split.html
    │   ├── Error Metrics.xlsx
    │   ├── Model Application-CNN.html
    │   ├── Model Application-KNN.html
    │   ├── Model Application-Random Forest.html
    │   ├── Predict All Models-Prediction Set.html
    │   ├── Predict CNN-Prediction Set.html
    │   ├── Predict CNN-Prediction Set_files
    │   ├── Predict CNN-Test Dataset.html
    │   ├── Predict KNN-Prediction Set.html
    │   ├── Predict KNN-Test Dataset.html
    │   ├── Predict Random Forest-Prediction Set.html
    │   ├── Predict Random Forest-Test Dataset.html
    │   ├── Prediction Set-All Pairs(9) Trajectories for All models.html
    │   ├── Save Model for various Reaction Times-KNN.html
    │   ├── Save Model for various Reaction Times-Neural.html
    │   ├── Save Model for various Reaction Times-Random Forest.html
    │   ├── Visualize Prediction Sets.html
    │   └── Visualize Test Sets.html
    ├── scripts
    │   ├── Calculate R2 and RMSE.ipynb
    │   ├── Cleaned Data set Stats.ipynb
    │   ├── Data-Cleanup_Transformation_Test_Train_Split.ipynb
    │   ├── Ethics_Checklist_Group5_Section03.ipynb
    │   ├── Explanatory Plot ppt.ipynb
    │   ├── Model Application-CNN.ipynb
    │   ├── Model Application-KNN.ipynb
    │   ├── Model Application-Random Forest.ipynb
    │   ├── Predict All Models-Prediction Set.ipynb
    │   ├── Predict All Models-Test Set.ipynb
    │   ├── Predict CNN-Prediction Set.ipynb
    │   ├── Predict CNN-Test Dataset.ipynb
    │   ├── Predict KNN-Prediction Set.ipynb
    │   ├── Predict KNN-Test Dataset.ipynb
    │   ├── Predict Random Forest-Prediction Set.ipynb
    │   ├── Predict Random Forest-Test Dataset.ipynb
    │   ├── Prediction Plots_0.1 for ppt.ipynb
    │   ├── Prediction Plots_0.2 for ppt.ipynb
    │   ├── Prediction Plots_0.3 for ppt.ipynb
    │   ├── Prediction Plots_0.5 for ppt.ipynb
    │   ├── Prediction Plots_1 for ppt.ipynb
    │   ├── Save Model for various Reaction Times-KNN.ipynb
    │   ├── Save Model for various Reaction Times-Neural.ipynb
    │   ├── Save Model for various Reaction Times-Random Forest.ipynb
    │   ├── Visualize Prediction Sets.ipynb
    │   └── Visualize Test Sets.ipynb
    └── src
        ├── Cleanup.py
        ├── EDA.py
        ├── FileProcessing.py
        ├── ModelClass.py
        └── Tranformation.py
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  The dataset used in this project is called * Next Generation Simulation (NGSIM) * and it is publicly available on the U.S. Department of Transportation website. The NGSIM dataset consists of 25 columns and 11.8 million rows of vehicle trajectory data which was captured using a network of synchronized digital video cameras on 4 different locations (US 101, I-80, Peachtree, and Lankershim). The data contains the location of a vehicle at every one-tenth of a second, which gives the exact position of each vehicle relative to other vehicles. 
  A total of 3 different types of vehicle data can be found in this dataset, namely Car, Truck, and Motorcycle. Most of the data were taken from the two freeways i.e., US 101 and I-80, and among the three vehicle types, data on Cars is more as compared to Trucks and Motorcycles. Therefore, we decided to work on only the Freeways. After the Preliminary analysis, we found that some vehicle IDs are present in more than one location meaning that the data from all four locations were taken separately and then merged in a single file. Hence, we separated the data based on location to carry out data cleaning and data transformation and then merge them back. 

<p align="center">
  <img src="images/NGSIM.png" alt="NGSIM.png" display="inline-block" width="80%" height="50%">
</p>

<p align="center">
  <img src="images/highways.png" alt="highways.png" display="inline-block" width="70%" height="70%">
</p>

 _The NGSIM dataset is publicly available. Please refer to the [Link](https://www.opendatanetwork.com/dataset/data.transportation.gov/8ect-6jqj)_ 

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ROADMAP -->
<h2 id="roadmap"> :dart: Roadmap</h2>

<p align="justify"> 
  We trained three models namely k-Nearest Neighbors, Random Forest, and CNN to predict Acceleration for Subject Vehicle and find rest of the trajectory.
  The goals of this project include the following:
<ol>
  <li>
    <p align="justify"> 
      Clean and Transform the NGSIM Data 
    </p>
  </li>
  <li>
    <p align="justify"> 
      Train Random Forest, KNN and CNN on same Train Data. 
    </p>
  </li>
  <li>
    <p align="justify"> 
      Compare R2 and RMSE for the models. 
    </p>
  </li>
  <li>
    <p align="justify"> 
      Validate Vehicle Trajectories and compare with R2 and RMSE outcomes.
    </p>
  </li>
</ol>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREPROCESSING -->
<h2 id="preprocessing"> :hammer: Preprocessing</h2>

<p align="justify"> 
  The NGSIM (Next Generation Simulation) dataset includes vehicle trajectory details along with the information of Lead and Following vehicle IDs. Even though the dataset has been processed by the Transportation department there are quite a few instances of Bad Data and vehicle trajectories which wont help in training for a good Car following model. Thus, we have preprocessed the data as per below:
  
<p align="center">
  <img src="images/Preprocessing.png" alt="Preprocessing.png" display="inline-block" width="80%" height="100%">
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PRE-PROCESSED DATA -->
<h2 id="Data-Cleanup"> :diamond_shape_with_a_dot_inside: Data-Cleanup</h2>

<p align="justify"> 
  Below are the cleanup tasks that have been performed:

   - Remove all Vehicles which have bad Vehicle Length and Type. i.e. same vehicle having two vehicle Types in different pair information.
   - Remove first 5 seconds for a Pair when Lead Vehicle is overtaking. Last 5 from the pair which the vehicle is leaving(same vehicle was Subject in that scneario).
   - Remove Lane 4 and above as they are shoulders/ramps and exits. 
   - Remove trajectories which are less than 30 seconds. 
   - Remove vehicles with Time headway greater than 10 seconds. 

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2 id="Data-Transormation"> :large_orange_diamond: Data Transformation</h2>

<p align="justify"> 
  Below are the transformation steps that have been followed:

   - Map Subject and Lead Vehicle information to have them on the same row. 
   - Convert the Feet information to Metric Metres.
   - Convert the Front - Front Space and Time Headway details to Lead Vehicle Rear Bumper to Subject Vehicle Front Bumper. 
   - Create per pair timing. 
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- RESULTS AND DISCUSSION -->
<h2 id="results-and-discussion"> :mag: Results and Discussion</h2>

<p align="justify">
  The overall accuracy score of personal and impersonal models are shown in the following tables. Some of the results we observed are similar to the results obtained by Weiss et.al and they are discussed below: <br>
</p>
<p align="justify">
<ul>
  <li>
    Since accelerometers senses acceleration based on vibration which can be more prominent during an activity and gyroscope only senses rotational changes, accelerometers outperformed gyroscope in all our models. <br>
  </li>
  <li>
    As the style of performing an activity differs from each person, it is difficult to aggregate those features among all subjects. So our personal models vastly outperformed our impersonal models.
  </li>
  <li>
    It is also observed that non hand-oriented activities are classified better with sensors from smartphone and handoriented activities are classified better with sensors from smartwatch. Refer appendix for activity wise recall scores. Some key take-aways based on our results are listed below:
  </li>
  <li>
    CNN trained on raw sensor data performed better for personal models, however it performed poorly on impersonal models.
  </li>
</ul>
</p>

<p align="center">
  <img src="images/Personal and Impersonal Table.png" alt="Table 3 and 4" width="75%" height="75%">
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- REFERENCES -->
<h2 id="references"> :books: References</h2>

<ul>
  <li>
    <p>Matthew B. Kennel, Reggie Brown, and Henry D. I. Abarbanel. Determining embedding dimension for phase-space reconstruction using a geometrical construction. Phys. Rev. A, 45:3403–3411, Mar 1992.
    </p>
  </li>
  <li>
    <p>
      L. M. Seversky, S. Davis, and M. Berger. On time-series topological data analysis: New data and opportunities. In 2016 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1014–1022, 2016.
    </p>
  </li>
  <li>
    <p>
      Floris Takens. Detecting strange attractors in turbulence. In David Rand and Lai-Sang Young, editors, Dynamical Systems and Turbulence, Warwick 1980, pages 366–381, Berlin, Heidelberg, 1981. Springer Berlin Heidelberg.
    </p>
  </li>
  <li>
    <p>
      Guillaume Tauzin, Umberto Lupo, Lewis Tunstall, Julian Burella P´erez, Matteo Caorsi, Anibal Medina-Mardones, Alberto Dassatti, and Kathryn Hess. giotto-tda: A topological data analysis toolkit for machine learning and data exploration, 2020.
    </p>
  </li>
  <li>
    <p>
      G. M. Weiss and A. E. O’Neill. Smartphone and smartwatchbased activity recognition. Jul 2019.
    </p>
  </li>
  <li>
    <p>
      G. M. Weiss, K. Yoneda, and T. Hayajneh. Smartphone and smartwatch-based biometrics using activities of daily living. IEEE Access, 7:133190–133202, 2019.
    </p>
  </li>
  <li>
    <p>
      Jian-Bo Yang, Nguyen Nhut, Phyo San, Xiaoli li, and Priyadarsini Shonali. Deep convolutional neural networks on multichannel time series for human activity recognition. IJCAI, 07 2015.
    </p>
  </li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> :scroll: Contributors</h2>

<p>
  :mortar_board: <i>All participants in this project are graduate students in the <a href="https://www.concordia.ca/ginacody/computer-science-software-eng.html">Department of Computer Science and Software Engineering</a> <b>@</b> <a href="https://www.concordia.ca/">Concordia University</a></i> <br> <br>
  
  :woman: <b>Divya Bhagavathiappan Shiva</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>divya.bhagavathiappanshiva@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/divyabhagavathiappan">@divyabhagavathiappan</a> <br>
  
  :woman: <b>Reethu Navale</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>reethu.navale@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/reethunavale">@reethunavale</a> <br>

  :woman: <b>Mahsa Sadat Afzali Arani</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>m_afzali93@yahoo.com</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/MahsaAfzali">@MahsaAfzali</a> <br>

  :boy: <b>Mohammad Amin Shamshiri</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>mohammadamin.shamshiri@mail.concordia.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/ma-shamshiri">@ma-shamshiri</a> <br>
</p>

<br>
✤ <i>This was the final project for the course COMP 6321 - Machine Learning (Fall 2020), at <a href="https://www.concordia.ca/">Concordia University</a><i>
