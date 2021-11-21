# DS_Breast_Cancer_Detection

Business Problem: 
1. Traditional models  are lacking in accuracy and do affect the overall patient diagnosis. Gail model used simple statistical architectures and the additional inputs were derived from costly and/or invasive procedures. 
2. The cost of gathering the patient data from EHR systems like EPIC and Cerner is very high and requires a lot of compliance clearance. Lack of centralized analytical platform that can generate the required data for breast cancer detection and provide results instantly

Goal:
1. Decrease the turnaround time in cancer detection process within patient diagnosis. 
2. Reduce the cost of generating the models by leveraging just the oncology data & Increase Insights for cohort treatment

Solution Approach: 
1. Explore the dataset and summarize the relations between the variables. Create  descriptive statistics, box plots, scatter plots, and a correlation matrix for all variables.
2. Use logistic regression to predict the diagnosis of breast tissue based on all or selected cell nucleus characteristics.
3. Build a classification model (single and pruned decision trees, random forest) and extract the rules to be used for the prediction of breast tissue diagnosis.
