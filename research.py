import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
import copy

class researcher:
    def __init__(self, data, model, train_loop, config, grid_search=True, create_model = None, max_trains=100, models = [], research_type = None):

        # data = self.dict_to_namespace(data)
        # config = self.dict_to_namespace(config)

        MyDynamicClass = type('obj', (object,), {})

        # set default train case
        self.train_case = None

        # set model
        self.model = model
        self.models = models
        self.create_model = create_model
        
        if max_trains is not None:
            self.max_trains = max_trains
        else:
            self.max_trains = 100

        # set inti now and config
        self.now = MyDynamicClass()
        self.now.hyperparameter = MyDynamicClass()
        self.now.model = MyDynamicClass()
        self.now.data = MyDynamicClass()
        self.now.epochs = None

        self.train_data = MyDynamicClass()

        self.config = MyDynamicClass()

        self.config.hyperparameter = MyDynamicClass()

        # self.config.hyperparameter.range = (getattr(config.hyperparameter, "range", None))
        # self.config.hyperparameter.choice = (getattr(config.hyperparameter, "choice", None))
        # self.config.hyperparameter.fixed = (getattr(config.hyperparameter, "fixed", None))

        hp = config.get("hyperparameter", {})
        self.config.hyperparameter.range = hp.get("range", {})
        self.config.hyperparameter.choice = hp.get("choice", {})
        self.config.hyperparameter.fixed = hp.get("fixed", {})

        if getattr(config, "model", None):
            self.config.model = MyDynamicClass()
            
            # self.config.model.range = getattr(config.model, "range", None)
            # self.config.model.choice = getattr(config.model, "choice", None)
            # self.config.model.fixed = getattr(config.model, "fixed", None)

            model_cfg = config.get("model", {})
            self.config.model.range = model_cfg.get("range", {})
            self.config.model.choice = model_cfg.get("choice", {})
            self.config.model.fixed = model_cfg.get("fixed", {})


        # set train type
        self.train_type = self.startGridSearchHyperparameter if grid_search else self.startRandomSearchHyperparameter
        self.model_train_type = self.startGridSearchModel if grid_search else self.startRandomSearchModel

        self.train_loop = train_loop

        # config now hyperparameter
        for [key,value] in self.config.hyperparameter.range.items():
            setattr(self.now.hyperparameter, key, value[0])

        for [key,value] in self.config.hyperparameter.choice.items():
            setattr(self.now.hyperparameter, key, value[0])

        for [key,value] in self.config.hyperparameter.fixed.items():
            setattr(self.now.hyperparameter, key, value)

        if getattr(self.config, "model", None):
            # config now model
            for [key,value] in self.config.model.range.items():
                setattr(self.now.model, key, value[0])

            for [key,value] in self.config.model.choice.items():
                setattr(self.now.model, key, value[0])

            for [key,value] in self.config.model.fixed.items():
                setattr(self.now.model, key, value)

        # config now data
        for [key,value] in data.items():
            setattr(self.now.data, key, value)

        # set research type
        if research_type == 1:
            self.train_case = "hyperparameter"
        elif research_type == 2:
            self.train_case = "hyperparameterAndConfigModel"
        elif research_type == 3:
            self.train_case = "hyperparameterAndChoiceModel"
        else:
            raise ValueError("unknown research_type: ", research_type)

        # default epochs
        if self.now.hyperparameter.epochs == None:
            self.now.hyperparameter.epochs = 10

        # set save model
        self.saveModel = {
            "epoch": None,
            "loss_train": torch.tensor(float('inf'), dtype = torch.float32),
            "loss_test": torch.tensor(float('inf'), dtype = torch.float32),
            "param": None,
            "info": "None",
        }

        print("Hyperparameter Researcher initialized.")
        for attr, value in self.now.hyperparameter.__dict__.items():
            print(f"  {attr}: {value}")

        if getattr(self.config, "model", None):
            print("Model Researcher initialized.")
            for attr, value in self.now.model.__dict__.items():
                print(f"  {attr}: {value}")

    def prepare_data_and_train_loop(self):
        self.train_data.data = self.now.data
        for attr, value in self.now.hyperparameter.__dict__.items():
            setattr(self.train_data, attr, value)

        self.train_data.model = copy.deepcopy(self.model)

        print("Prepared training data and hyperparameters for training loop.")
        for attr, value in self.train_data.__dict__.items():
            if attr != "data" and attr != "model":
                print(f"  {attr}: {value}")

        self.train_data.bestModel = self.bestModel
        self.train_data.accuracy_fn = self.accuracy_fn

        self.train_loop(self.train_data)

    def accuracy_fn(self, y_true,y_pred):
        y_pred = torch.argmax(y_pred, dim=1)
        correct = torch.eq(y_true,y_pred).sum().item()
        acc = (correct/len(y_pred))*100
        return acc


    def class_to_dict(self, obj):
        if isinstance(obj, dict):
            return {k: self.class_to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return {k: self.class_to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    def dict_to_namespace(self,d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self.dict_to_namespace(v) for k, v in d.items()})
        return d
        
    def startResearch(self):
        if self.train_case == "hyperparameter":
            print( "Starting hyperparameter research... with train type:", self.train_type )
            self.train_type()
        elif self.train_case == "hyperparameterAndConfigModel":
            print( "Starting hyperparameter and config model research... with train type:", self.model_train_type )
            self.model_train_type()
        elif self.train_case == "hyperparameterAndChoiceModel":
            print( "Starting hyperparameter and choice model research... with train type:", self.choice_model_train_type )
            self.choice_model_train_type()
        else:
            raise ValueError("unknown train_case: ", self.train_case)

    def choice_model_train_type(self):
        for model in self.models:
            self.model = model
            self.train_type()

    # Model
    def startGridSearchModel(self):
        model_keys_range = list(self.config.model.range.keys())
        model_keys_choice = list(self.config.model.choice.keys())

        def grid_search_model_choice(index):
            if index == len(model_keys_choice):
                print("Training with model configuration:", self.now.model)
                self.model = self.create_model(self.now.model)
                self.train_type()
                return

            key = model_keys_choice[index]
            for value in self.config.model.choice[key]:
                setattr(self.now.model, key, value)
                grid_search_model_choice(index + 1)

        def grid_search_model_range(index):
            if index == len(model_keys_range):
                grid_search_model_choice(0)
                return

            key = model_keys_range[index]
            low, high, step = self.config.model.range[key]
            value = low
            while value <= high:
                setattr(self.now.model, key, value)
                grid_search_model_range(index + 1)
                value += step
        grid_search_model_range(0)

    def startRandomSearchModel(self):
        max_trains = self.max_trains
        model_keys_range = list(self.config.model.range.keys())
        model_keys_choice = list(self.config.model.choice.keys())

        for _ in range(max_trains):
            for key in model_keys_range:
                low, high, step = self.config.model.range[key]
                random_value = np.random.uniform(low, high)
                # Round the random value to the nearest step
                random_value = round((random_value - low) / step) * step + low
                setattr(self.now.model, key, random_value)

            for key in model_keys_choice:
                choice_values = self.config.model.choice[key]
                random_value = np.random.choice(choice_values)
                setattr(self.now.model, key, random_value)

            print("Training with model configuration:", self.now.model)
            self.model = self.create_model(self.now.model)
            self.train_type()

    # Hyperparameter
    def startGridSearchHyperparameter(self):
        hyper_keys_range = list(self.config.hyperparameter.range.keys())
        hyper_keys_choice = list(self.config.hyperparameter.choice.keys())

        def grid_search_hyperparameter_choice(index):
            if index == len(hyper_keys_choice):
                print("Training with hyperparameters:", self.now.hyperparameter)
                self.prepare_data_and_train_loop()
                return

            key = hyper_keys_choice[index]
            for value in self.config.hyperparameter.choice[key]:
                setattr(self.now.hyperparameter, key, value)
                grid_search_hyperparameter_choice(index + 1)

        def grid_search_hyperparameter_range(index):
            if index == len(hyper_keys_range):
                grid_search_hyperparameter_choice(0)
                return

            key = hyper_keys_range[index]
            low, high, step = self.config.hyperparameter.range[key]
            value = low
            while value <= high:
                setattr(self.now.hyperparameter, key, value)
                grid_search_hyperparameter_range(index + 1)
                value += step
        grid_search_hyperparameter_range(0)

    def startRandomSearchHyperparameter(self):
        max_trains = self.max_trains
        hyper_keys_range = list(self.config.hyperparameter.range.keys())
        hyper_keys_choice = list(self.config.hyperparameter.choice.keys())

        for _ in range(max_trains):
            for key in hyper_keys_range:
                low, high, step = self.config.hyperparameter.range[key]
                random_value = np.random.uniform(low, high)
                # Round the random value to the nearest step
                random_value = round((random_value - low) / step) * step + low
                setattr(self.now.hyperparameter, key, random_value)

            for key in hyper_keys_choice:
                choice_values = self.config.hyperparameter.choice[key]
                random_value = np.random.choice(choice_values)
                setattr(self.now.hyperparameter, key, random_value)

            print("Training with hyperparameters:", self.now.hyperparameter)
            self.prepare_data_and_train_loop()

    # Best Model
    def bestModel(self, loss_train, loss_test, model, epoch, info = None):

        if loss_test < self.saveModel["loss_test"]:
            self.saveModel["epoch"] = epoch
            self.saveModel["loss_train"] = loss_train
            self.saveModel["loss_test"] = loss_test
        
            
            torch.save(model.state_dict(), "model_weights.pth")
            self.saveModel["param"] = "model_weights.pth"

            torch.save(model, 'full_model.pth')
            self.saveModel["model"] = 'full_model.pth'

            if info is not None:
                self.saveModel["info"] = info

            return 1

        return 0

    # Load Model
    def loadModel(self, model, full=False):
        self.model.load_state_dict(torch.load(self.saveModel["param"]))
        if full:
            model = torch.load(self.saveModel["model"])
        return model
    
    # Print
    def print_info(self):
        print("Best Model Info:")
        print("Epoch:", self.saveModel["epoch"])
        print("Train Loss:", self.saveModel["loss_train"])
        print("Test Loss:", self.saveModel["loss_test"])
        print("Additional Info:", self.saveModel["info"])