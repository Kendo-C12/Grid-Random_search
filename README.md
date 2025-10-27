argunments explain:
  config: the data that will be use in train loop
    there will be 2 type of config 1.hyperparameter 2.model
      there will be 3 catagory include range -> [start,end,step] note that start have to be less than end
                                       choice -> list that contain all of choice in that hyperparameter
                                       fixed -> fixed hyperparameter can also write ass choice but contain 1 choice
  data -> data you use to train there should be in numpy array 
  train loop -> main loop that will start training note that class won't preprocess data you have to do it by yourself to                     call the data it have be inform of self."name of hyperparameter" and the data will be inform self.data."X/y                  train , X/ytest" the function must take self as argument the function don'y have to return anything since                    class didn't save these data
      the train loop should call these function when end of every epoch: 
      |
      |---> self.bestModel(avg_loss_train, avg_loss_test, model, epoch, info)
      the example will be in test.ipynb
  model -> model you use have to train in train loop you can access model be using self.model
  
  research_type:
    code        type
    1           only hyperparameter
    2           hyperparameter and model with config
    3           hyperparameter and model with choice
                                           

argunments required:
  public required:
    data: {
      X_train (numpy),
      y_train (numpy),
      X_test (numpy),
      y_test (numpy)
    }
    config: {
      hyperparameter: {
        range: dict,
        choice: dict,
        fixed: dict
      },
      model: {
        range: dict,
        choice: dict,
        fixed: dict
      }
    },
    train_loop: function,
    research_type: int (1, 2, 3)
      1: only hyperparameter
      2: hyperparameter and model with config
      3: hyperparameter and model with choice

  case research_type required:
    1:
      model is nn.module
    2:
      model is None
      create_model is function that returns nn.module
    3:
      model is None
      models is list of nn.module



  public optional:
    grid_search: bool (default: True)

  case research_type optional:
    1:
      pass
    2:
      pass
    3:
      max_trains: int (default: 100)

  
