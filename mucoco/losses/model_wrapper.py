class ModelWrapper:
    def __init__(self, model):
        self.__dict__["model"] = model
        self.__dict__["last_forward_step"] = -1
        self.__dict__["last_forward_result"] = None
    
    def forward(self, *args, **kwargs):
        step = kwargs.get("step", -1)
        if step != -1 and self.last_forward_step == step:
            # print("no need to go this again", step)
            return self.last_forward_result
        
        if step != -1:
            del kwargs["step"]
        
        self.last_forward_step = step
        go_inside = kwargs.get("go_inside", None)
        if go_inside is not None:
            func = getattr(self.model, go_inside)
            # print(x)
            del kwargs['go_inside']
            self.last_forward_result = func(*args, **kwargs)
        else:
            self.last_forward_result = self.model(*args, **kwargs)

        return self.last_forward_result
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def zero_grad(self, set_to_none=False):
        self.model.zero_grad(set_to_none=set_to_none)
        self.last_forward_step = -1
        self.last_forward_result = None

    def __getattr__(self, name):
        return getattr(self.model, name)
    
    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self.model.__setattr__(name, value)

