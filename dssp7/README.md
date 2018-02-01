## 1. new_project package
The default package contains a project that predict a rating score for a list of film.
### 1.1 Requirements
This project cannot be executed on sparrow. Your computer must have python 2.7 installed and all the packages in the `requirement.txt` .
### 1.2 Package installation
You are now going to install the **new_package**.
Open a terminal session and access the directory `new_project`.

- Install the package the default package :
```shell
make psi
make clean
```

You have completed the installation of this package.

### 1.3 Command line
#### 1.3.1 Predict score of film list
Enter the command below
```shell
proj-run-fit
proj-run-predict
```
The first command will launch the model fitting and save it as a pickled model stored in the `models` folder. This model is timestamped.

The second command will launch the prediction of the film list. It returns a dataframe stored in the `data/predicts` folder. We add `result` at the end of the input name to create this dataframe's name : `to_predict-result.csv`


(You are free to replace `to_predict.csv` by any csv containing the required information.)	