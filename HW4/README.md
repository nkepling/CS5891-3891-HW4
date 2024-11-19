# CS5891-3891-HW4

Follow the steps below to set up your Python environment for this assignment. You will need **python3.10.** 


## Create virtual environment. 

**venv**
```bash
python3.10 -m venv ns_gym_env
source ns_gym_env/bin/activate
```

**conda**
```bash
conda create -n ns_gym_env python=3.10
conda activate ns_gym_env
```


## Windows

**venv**
```powershell
python -m venv C:\path\to\new\virtual\environment
C:\path\to\ns_gym_env\Scripts\activate
```

**conda**
```cmd
conda create -n ns_gym_env python=3.10
conda activate ns_gym_env
```


## Install NS-Gym from source

```bash
pip install git+https://github.com/scope-lab-vu/ns_gym.git
pip install tqdm
```

With this installation you will have access to the following packages:

1. `numpy`
2. `gymnasium`
3. `ns_gym`

**Do not** pip install any other dependencies. You are free to use any functions with these libraries in a addition to those builtin to Python3.10.

