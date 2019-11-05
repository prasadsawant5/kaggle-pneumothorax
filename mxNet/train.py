from mxNet.architectures.my_model import MyModel
import mxnet as mx
    
class MxNetTrain:
    def optimize(self, net, optimizer='adam', learning_rate=0.1, ctx=mx.gpu()):
        # net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        # trainer = gluon.Trainer(net.collect_params(), 
        #                         optimizer, 
        #                         {'learning_rate': learning_rate})
        # return trainer
        pass

    def run(self):
        learning_rate = 1e-4
        
        myModel = MyModel()

        # net = myModel.fcn_model(kp=0.7)

        print(myModel)

        self.optimize(myModel, learning_rate=learning_rate)