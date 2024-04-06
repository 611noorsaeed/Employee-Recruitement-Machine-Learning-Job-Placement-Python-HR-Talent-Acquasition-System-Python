import pickle
from flask import request, Flask, render_template, redirect, url_for, flash
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# ======================================Create app=================================================
app = Flask(__name__)

# ======================================loading models and datasets================================
df = pd.read_csv('notebook/HR_comma_sep.csv.crdownload')
model = pickle.load(open('models/model.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

# ======================================dashboard functions========================================
def reading_cleaning(df):
    df.drop_duplicates(inplace=True)
    cols = df.columns.tolist()
    df.columns = [x.lower() for x in cols]

    return df
#-----
df = reading_cleaning(df)


def employee_important_info(df):
    # Average satisfaction level
    average_satisfaction = df['satisfaction_level'].mean()
    # Department-wise average satisfaction level
    department_satisfaction = df.groupby('department')['satisfaction_level'].mean()
    # Salary-wise average satisfaction level
    salary_satisfaction = df.groupby('salary')['satisfaction_level'].mean()

    # Employees who left
    left_employees = len(df[df['left'] == 1])
    # Employees who stayed
    stayed_employees = len(df[df['left'] == 0])

    return average_satisfaction, department_satisfaction, salary_satisfaction, left_employees, stayed_employees

def plots(df, col):
    values = df[col].unique()
    plt.figure(figsize=(15, 10))

    explode = [0.1 if len(values) > 1 else 0] * len(values)
    plt.pie(df[col].value_counts(), explode=explode, startangle=40, autopct='%1.1f%%', shadow=True)
    labels = [f'{value} ({col})' for value in values]
    plt.legend(labels=labels, loc='upper right', fontsize=12)
    plt.title(f"Distribution of {col}", fontsize=16, fontweight='bold')

    plt.savefig('static/'+ col + '.png')
    plt.close()

def distribution(df, col):
    values = df[col].unique()
    plt.figure(figsize=(15, 10))
    sns.countplot(x=df[col], hue='left', palette='Set1', data=df)
    labels = [f"{val} ({col})" for val in values]
    plt.legend(labels=labels, loc="upper right", fontsize=12)
    plt.title(f"Distribution of {col}", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90)
    plt.savefig('static/' + col + '_distribution.png')
    plt.close()

def comparison(df, x, y):
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y, hue='left', data=df, ci=None)
    plt.title(f'{x} vs {y}', fontsize=16, fontweight='bold')
    plt.savefig('static/' + 'comparison.png')
    plt.close()


def corr_with_left(df):
    df_encoded = pd.get_dummies(df)
    correlations = df_encoded.corr()['left'].sort_values()[:-1]
    colors = ['skyblue' if corr >= 0 else 'salmon' for corr in correlations]
    plt.figure(figsize=(15, 10))
    correlations.plot(kind='barh', color=colors)
    plt.title('Correlation with Left', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/correlation.png')
    plt.close()

def histogram(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # Create a grid of 1 row and 2 columns

    # Plot the first histogram
    sns.histplot(data=df, x=col, hue='left', bins=20, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}", fontsize=16, fontweight='bold')

    # Plot the second histogram
    sns.kdeplot(data=df, x='satisfaction_level', y='last_evaluation', hue='left', shade=True, ax=axes[1])
    axes[1].set_title("Kernel Density Estimation", fontsize=16, fontweight='bold')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig('static/' + col + '_histogram.png')
    plt.close()

#=====================prediction function====================================================
def prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    data = {
    'sl_no': [sl_no],
    'gender': [gender],
    'ssc_p': [ssc_p],
    'hsc_p': [hsc_p],
    'degree_p': [degree_p],
    'workex': [workex],
    'etest_p': [etest_p],
    'specialisation': [specialisation],
    'mba_p': [mba_p]
    }
    data = pd.DataFrame(data)
    data['gender'] = data['gender'].map({'Male':1,"Female":0})
    data['workex'] = data['workex'].map({"Yes":1,"No":0})
    data['specialisation'] = data['specialisation'].map({"Mkt&HR":1,"Mkt&Fin":0})
    scaled_df = scaler.transform(data)
    result = model.predict(scaled_df).reshape(1, -1)
    return result[0]


# routes===================================================================

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def home():
    return render_template("index.html")
@app.route('/job')
def job():
    return render_template('job.html')


@app.route('/ana')
def ana():
    average_satisfaction, department_satisfaction, salary_satisfaction, left_employees, stayed_employees= employee_important_info(df)
    plots(df, 'left')
    plots(df, 'salary')
    plots(df, 'number_project')
    plots(df, 'department')

    distribution(df, 'salary')
    distribution(df, 'department')

    comparison(df, 'department', 'satisfaction_level')

    corr_with_left(df)

    histogram(df, 'satisfaction_level')

    # Convert Series objects to dictionaries
    department_satisfaction= department_satisfaction.to_dict()
    salary_satisfaction = salary_satisfaction.to_dict()
    return render_template('ana.html', df=df.head(),average_satisfaction=average_satisfaction,
                           department_satisfaction=department_satisfaction,salary_satisfaction=salary_satisfaction,
                           left_employees=left_employees,stayed_employees=stayed_employees)



#prediction===============================================================
@app.route("/placement",methods=['POST','GET'])
def placement():
    if request.method == 'POST':
        sl_no = request.form['sl_no']
        gender = request.form['gender']
        ssc_p = request.form['ssc_p']
        hsc_p = request.form['hsc_p']
        degree_p = request.form['degree_p']
        workex = request.form['workex']
        etest_p = request.form['etest_p']
        specialisation = request.form['specialisation']
        mba_p = request.form['mba_p']

        result = prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)

        if result == 1:
            pred = "Placed"
            rec = "We recommend you that this is the best candidate for you business"
            return render_template('job.html', result=pred, rec=rec)

        else:
            pred = "Not Placed"
            rec = "We recommend you that this is not the best candidate for your business"
            return render_template('job.html', result=pred,rec=rec)

    return redirect(url_for('index'))

# ========================python main===================================================
if __name__ == "__main__":

    app.run(debug=True)
