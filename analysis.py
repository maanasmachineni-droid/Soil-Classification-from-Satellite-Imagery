import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')
try:
    df = pd.read_csv('performance.csv')

    print("\n---Raw Data---")
    print(df.head())

    print("\n---Statistical Summary---")
    print(df.describe())

    scores=df[["Math", "Science", "English"]].to_numpy()

    mean_scores = np.mean(scores, axis=0)
    median_scores = np.median(scores, axis=0)  
    std_dev_scores = np.std(scores, axis=0)

    print("\n---NumPy Analysis---")
    print("Mean Scores(Math, Science, English):", mean_scores)
    print("Median Scores(Math, Science, English):", median_scores)
    print("Standard Deviation of Scores(Math, Science, English):", std_dev_scores)
    

    top_math=df.loc[df["Math"].idxmax(), "Name"]
    top_science=df.loc[df["Science"].idxmax(), "Name"]
    top_english=df.loc[df["English"].idxmax(), "Name"]

    print("\n---Top Performers---")
    print(f"Top in Math: {top_math}")
    print(f"Top in Science: {top_science}")
    print(f"Top in English: {top_english}")

    subject=["Math", "Science", "English"]
    plt.bar(subject, mean_scores, color=['blue', 'orange', 'green'])
    plt.title("Average Scores by Subject")
    plt.xlabel("Subjects")
    plt.ylabel("Average Score")
    plt.show()

    df.plot(x="Name", y=["Math", "Science", "English"], kind="bar")
    plt.title("Scores of Students in Different Subjects")
    plt.xlabel("Students")
    plt.ylabel("Scores")
    plt.show()
except FileNotFoundError:
    print("Error: 'performance.csv' file not found. Please ensure the file is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")