# DAPT-619 Assignment 1
## Due date: November 11th, 2025 11:59 PM

### Setup
1. Open the assignments repo by clicking on this link: https://github.com/vishal-git/dapt-619-assignments 
2. Click **Use this template** to create a new repository. In the form:  
    a) Owner: pick *your* GitHub account.  
    b) Repository name: `dapt-619-hw1-<lastname>` (e.g., `dapt619-hw1-patel`)  
    c) Description: optional  
    d) Visibility: Private or Public  
    e) Include all branches: Leave unchecked.   
Click **Create repository from template**.  
On the new repo page, click **Code -> HTTPS** and copy the URL.  
In a terminal run the following:  
```
git clone https://github.com/<your-username>/dapt-619-<lastname>.git
cd dapt-619-<lastname>
```
3. Create a **feature branch** for your work. 
4. Add a `README.md` file using the `touch` command in your terminal. Add your first and last name to this file using the `echo` command with `>`. Also append the current date to the file. (The exact date doesn't matter.)
5. Create a **Python virtual environment** for your work. Activate it. [You can verify that you're using the Python executable from your virtual environment by running `which python` or `where python`.]
6. Install all Python dependencies using the provided `requirements.txt`.

### Scoring Script
7. A Linear Regression model has been trained using `eruptions` as the *independent* variable and `waiting` as the *dependent* (target) variable. Review the code in `./src/simple_linear_demo.py`. The saved model is available in the `./models/` directory.
Write a scoring script that reads the Geyser dataset, scores it using the model artifact, and exports the scored dataset to a new `./data/scored/` folder.
8. Check which files are being tracked using `git status`, and then add the scoring script to Git using `git add`.
9. If the Python virtual environment folder (e.g., `venv`) shows up as a tracked folder, follow these steps:  
    a) Create a file named `.gitignore`.  
    b) Add that folder name to it using the `echo` command.   
    c) If you run `git status` again, the virtual environment folder won't show up as a tracked file.  
    d) Stage the `.gitignore` file using `git add`.  
10. Run this script with the `python` command and export the scored dataset with three columns: two columns from the original dataset and one additional column that contains the predictions.
11. Add the scored dataset to Git.
12. Commit your changes to the Git repository. Write a meaningful commit message. This commit should include the scoring script, the scored dataset, and the `.gitignore` file (if created in step 9). 
13. Push your changes to the remote branch.

### Plotting
14. Modify your scoring script to add some code that creates a plot between actual vs. predicted waiting times. Use your choice of Python package for plotting. Install the package of your choice and update `requirements.txt` file accordingly.[Hint: Use `pip freeze`.]
15. Save the figure as `geyser_predictions.png` in a `plots/` folder.
16. Commit and push your changes to the remote branch. 

### Unit Testing
17. Write a test function to ensure the scoring process works correctly. Include three separate `assert` statements:  
    a) The number of predicted values must match the number of input values for `eruptions`.  
    b) All predicted values must be finite (not NaN or inf).  
    c) All predicted values must be positive (greater than zero).  
Write a single unit test function that includes these checks. Save the unit test as `test_scoring.py` inside a `tests/` folder.
Use the three values provided in the code below for these checks.
```
    pipeline = joblib.load(model_path)

    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    preds = pipeline.predict(df[["eruptions"]])
```
18. Install `pytest` in your Python virtual environment.
19. Update `requirements.txt` to include `pytest`. 
20. Run the tests with `pytest` and make sure the test passes. Note: I only need the unit test code; I do not need to see the output. I will run the tests on my computer to verify that they work as expected.
21. Push all files created from steps 17 through 21 to the remote branch.

Finally, don't forget to `deactivate` your Python virtual environment!

Here's a short submission checklist for your reference:  
[ ] `README.md` includes name and date  
[ ] Virtual environment created and activated  
[ ] Scoring script runs without error  
[ ] Scoring outputs (scored dataset and plot) are generated  
[ ] Unit test file created and passes all tests  
[ ] All changes (except for the virtual environment folder) committed and pushed to remote branch  
