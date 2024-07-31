# Solving High Attrition Rate Problem

## Business Understanding

Vidtribe has more than 1000 employees, sprearheaded across Indonesia. Even though it has become a fairly large company, Vidtribe still has quite a lot of difficulties in managing employees. This has an impact on the high attrition rate (the ratio of the number of employees who leave to the total number of employees) up to more than 10%.

To prevent this from getting worse, HR department managers need help in identifying the various factors that affect the high attrition rate. In addition, it also need to monitor the respective factor by creating a dashboard. After insights found, recommend action item so that the company can lowering the attrition rate by paying attention to the root factor found.

### Business Problem

The problem Vidtribe trying to solve is the high number of attrition rate, they need to identify what causes the problem and then need to monitor it via dashboard.

### Project Scope

This project will be focus on two things: extracting insights from the given dataset to draw action item so that the company can reduce the high attrition rate, and build a machine learning model to predict whether the employee will be stayed or resigned. The machine learning model will not be deployed, but python script for making prediction will be available.

### Steps

1. Data wrangling including data gathering, assesing data, and cleaning the data
2. Ask initial questions, then answer it by creating charts which then will be used to create monitoring dashboard
3. Exploratory data analysis by conducting bivariate and multivariate analysis
4. Model preparation, by encoding categorical features and drop features with very strong correlation to avoid redundancy
5. Model training, by utilizing boosting and tree-based algorithms to find the best performing model with the highest precision score
6. Model evaluation, make used of hyperparameter tuning for the best performing model discovered so that precision score can increase 

### Preparation

[Click to see the data source](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee). We need to use jupyter notebook for data exploration and model experimentation, then utilized Tableau Public to create dashboard.

## Initial Questions
Here's the initial questions that will be answered in dashboard:

1. How many hired employees?, how many active employees and resign employees?
2. How's the attrition rate?
3. How's the number of employee by service year?
4. How's the number of employee by age group (18-25, 26-35, 36-45, 46-55, 56-65, Over 65)? 
5. How's the number of employee by satisfaction rate?
6. How's the attrition rate by department for each service year group?
7. How's the resignation number by job role?
8. How's the proportion of gender attrition rate for different age group?
10. How's the number of resignation by age group (18-25, 26-35, 36-45, 46-55, 56-65, Over 65)?
11. How's the proportion of overtime in resigned employees?
8. How's the number of resigned employee according to business travel?
9. How's the performance rating of resigned employees?
10. How's the work life balance of resigned employees?
11. How's the job satisfaction rating of resigned employees?

## Business Dashboard

The dashboard contains of 3 pages: Hired Employee, Resignment Summary, and Resignment in Detail. In each page, user can filter the chart based on gender, education field, and martial status. [Here's the link to access the dashboard.](https://public.tableau.com/views/HRAttritionDashboard_17205009591570/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

![Roissyah Fernanda- dashboard page 1](https://github.com/user-attachments/assets/897ff785-6e07-47c0-98b0-0e208103ef3e)
* The first page deep dive into important KPIs in monitoring attrition, includes number of hires (active employees and resigned employees), active employee, resign employee, and attrition rate which was calculated by dividing the number of employee who left with the number of all employee times by 100%. The first page allow user to monitor the number of hire by several factor, includes job role, service years, age group, and job satisfaction rating.

![Roissyah Fernanda- dashboard page 2](https://github.com/user-attachments/assets/5d2ac2be-0690-4523-9a96-d132e25b1b7d)
* The second page focus on monitoring employee who left the company. There is a table to better understand about attrition rate in each department, grouped by its service year. User also allowed to monitor the resignation number across all job role. Lastly, user can derive insights about attrition rate proportion in each gender for different age group. 

![Roissyah Fernanda-dashboard page 3](https://github.com/user-attachments/assets/cc913ef6-9d2f-44b9-b8c9-106a3f6164be)
* The third page offers easy access to better understand the resignment in detail. There are five charts and one table, to explore number of resignation by age group, and by business travel. User can also draw insights from 4 factors that contribute to resignment: overtime, performance rating, work life balance, and job satisfaction rating.

## Insights discovered
Here's the **important information gained**:
1. Vidtribe has 1058 employees, but 16.919% of them were resigned resulting in high attrition rate. 
2. Sales Excecutive and Research Scientist appear to be the most in-demand roles, with the highest number of hires.
3. The majority of hires are between the ages of 26 and 45.
4. Laboratory Technician have the highest number of resignations (49), followed by Sales Excecutive (39) and Research Scientist (25).
5. Attrition rates are generally higher for younger employees (18-35) compared to older employees. There seems to be a particularly high attrition rate for men aged 26-35 (26.82%).


Here's the **valuable insights found**:
1. Attrition rates are consistently higher for employees in their first five years of service compared to those with more than five years of service across all departments. It suggests that Vidtribe might have a problem retaining employees during their first five years. There could be factors related to onboarding, training, or compensation that are leading new hires to leave the company.
2. The majority of resignations (51%) come from employees aged between 26 and 35 years old. While unexpected, this could be due to a number of reasons, including: older workers may be planning retirement and looking to reduce their workload, there could be a lack of career development opportunities for older employees, the company culture may not be a good fit for older workers.
3. Job role: Laboratory Technician, Sales Excecutive, and Research Scientist with the top 3 number of hires are also the top 3 job roles with the high number of resignation. Despite these 3 roles has the highest attrition rate, data shows that performance rating, work life balance, and job satisfaction majority categorized as excellent.
4. The age groups with the highest attrition rates are also the ones with the highest number of hires.
5. When the data is segmented by department, there's a gender disparity in attrition rate especially for service year of 0-5 and 6-10. The number of resigned male employee's is higher than the number of resigned female employees.
6. It appears that a higher percentage of resigned employees (54.75%) worked overtime compared to those who didn't (45.25%). This is surprising, as overtime is often seen as a factor contributing to employee burnout and turnover. It might be that employees with higher workloads are more valued by the company and are offered incentives to stay.
7. The majority of resigned employees (51%) rarely traveled for business. Similar to the overtime data, this challenges the notion that business travel is a major factor in employee attrition.
8. The majority of resigned employees (67%) had performance ratings exceeding expectations. This suggests that high performers are leaving the company, which is a significant concern. There could be several reasons for this: employees may feel under-compensated or under-challenged, there may be a lack of growth opportunities within the company, more tenured employees may be more likely to find better opportunities elsewhere.
9. The majority of resigned employees (42%) reported high job satisfaction, and (52.514%) reported having excellent work life balance This suggests that a significant number of employees are leaving despite reporting high satisfaction and excellent work life balance. This could be due to factors outside of the job itself, such as: compensation and benefits, lack of work-life balance due to personal reasons, commute.

## Conclusion

1. It was found that factors contribute to high number of attrition rate in Vidtribe are: OverTime, Age group, Service year, Job role, Department, Performance rating, and Job satisfaction rating. 
2. From the experimentation, the best model to predict attrition is decision tree, with recall score in class 1 is 0.88, recall score in class 1 is 0.38, and the ROC area is 0.63. Over 15 features used as predictor, the top 5 most importance variables of the model are: OverTime, StockOptionLevel, MonthlyRate, Age, and Job Satisfaction.

## Action item recommendation
1. Investigate why a high number of employees, particularly Sales Representatives and younger employees especially men aged 26-35 are leaving within the first five years. This could be due to factors like lack of work-life balance, career development opportunities, or parental leave policies. Focus on improving the onboarding process, providing adequate training, and offering competitive compensation and benefits packages for new hires.
2. Implement a stay interview program to gather feedback from employees after they have been with the company for a while. This can help identify areas for improvement that could reduce attrition among tenured employees.
3. While promising, the data about job stisfaction, performance rating, work life balance, and overtime only reflects employees' in initial impressions. It would be helpful to track how satisfaction levels change over time, especially among employees who are more likely to leave within the first five years
4. Conduct exit interviews with employees who are resigning especially those who are high performers, reported having high satisfaction and excellent work life balance, to understand their reasons for leaving. This can help identify specific issues related to certain job roles and departments.
5. Investigate why a higher percentage of employees who did not work overtime resigned. It could be due to factors such as: underutilization of skills and boredom, lack of work-life balance due to other reasons (e.g., long commute).

## Steps to run the prediction script
**⚒️ Setup environment**
```
conda create --name attrition-prediction python=3.9
conda activate attrition-prediction
pip install pickle-mixin pandas scikit-learn
```

**🚀 Run python script**
```
python run prediction.py
```