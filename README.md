ios-BPN-Algorithm
=================

This algorithm used EBP is one kind of Back Propagation Neural Networks ( BPN ), and also it is one of neural-network algorithms.

``` objective-c
#import "KRBPN.h"

@interface ViewController ()

@property (nonatomic, strong) KRBPN *_krBPN;

@end

@implementation ViewController

@synthesize _krBPN;

- (void)viewDidLoad
{
    [super viewDidLoad];
    
	_krBPN = [KRBPN sharedNetwork];
    
    //各輸入向量陣列值
    _krBPN.inputs = [NSMutableArray arrayWithObjects:
                     //Input Pattern 1
                     @[@1, @2, @0.5, @1.2],
                     //Input Pattern 2
                     @[@0, @1, @0.3, @-0.9],
                     //Input Pattern 3
                     @[@1, @-3, @-1, @0.4],
                     nil];
    /*
     * @ 輸入層、隱藏層、輸出層之間的神經元初始權重
     *
     *   - W14 : 輸入層 X1 到隱藏層 Net 4
     *   - W15 : 輸入層 X1 到隱藏層 Net 5
     *
     *   - W24 : 輸入層 X2 到隱藏層 Net 4
     *   - W25 : 輸入層 X2 到隱藏層 Net 5
     *
     *   - W34 : 輸入層 X3 到隱藏層 Net 4
     *   - W35 : 輸入層 X3 到隱藏層 Net 5
     *
     *   - W44 : 輸入層 X4 到隱藏層 Net 4
     *   - W45 : 輸入層 X4 到隱藏層 Net 5
     *
     *   - W46 : 隱藏層 Net 4 到輸出層的 Net 6
     *   - W56 : 隱藏層 Net 5 到輸出層的 Net 6
     *
     */
    //輸入層各向量值到隱藏層神經元的權重 ( 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組 )
    _krBPN.inputWeights  = [NSMutableArray arrayWithObjects:
                            //W14, W15
                            @[@0.2, @-0.3],
                            //W24, W25
                            @[@0.4, @0.1],
                            //W34, W35
                            @[@-0.5, @0.2],
                            //W44, W45
                            @[@-0.1, @0.3],
                            nil];
    //隱藏層神經元的偏權值
    _krBPN.hiddenBiases  = [NSMutableArray arrayWithObjects:
                            //Net 4
                            @-0.4,
                            //Net 5
                            @0.2,
                            nil];
    //隱藏層神經元到輸出層神經元的權重值
    _krBPN.hiddenWeights = [NSMutableArray arrayWithObjects:
                            //W46
                            @-0.3,
                            //W56
                            @-0.2,
                            nil];
    //有幾顆隱藏層的神經元 ( 不用外部設定，由偏權值數目自動設定 )
    //_krBPN.countHiddens;
    //輸出層神經元偏權值, Net 6 for output
    _krBPN.outputBias       = 0.1f;
    //期望值
    _krBPN.targetValue      = 1.0f;
    //學習速率
    _krBPN.learningRate     = 0.8f;
    //收斂誤差值 ( 一般是 10^-3 或 10^-6 )
    _krBPN.convergenceError = 0.001f;
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //每一次的迭代( Every generation-training )
    [_krBPN setEachGeneration:^(NSInteger times, NSDictionary *trainedInfo){
        NSLog(@"Generation times : %i", times);
        //NSLog(@"trainedInfo : %@\n\n\n", trainedInfo);
    }];
    
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes) {
        if( success )
        {
            if( !_weakKrBPN.trainedNetwork )
            {
                [_weakKrBPN saveTrainedNetwork];
            }
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo : %@", trainedInfo);
            NSLog(@"TrainedNetwork with inputWeights : %@\n\n\n", [_weakKrBPN.trainedNetwork.inputWeights description]);
        }
    }];
    
    //Remove your testing trained-network records.
    [_krBPN removeTrainedNetwork];
    
    //Start the random weights, biases
    [_krBPN trainingWithRandom];

    //Start the training network, and it won't be saving the trained-network when finished.
    [_krBPN training];
    
    //Start the training network, and it will auto-saving the trained-network when finished.
    [_krBPN trainingDoneSave];
    
    //If you wanna pause the training.
    [_krBPN pause];
    
    //If you wanna continue the paused training.
    [_krBPN continueTraining];
    
    //If you wanna reset the network back to initial situation.
    [_krBPN reset];
    
    //When the training finished, to save the trained-network into NSUserDefaults.
    [_krBPN saveTrainedNetwork];
    
    //If you wanna recover the trained-network data.
    [_krBPN recoverTrainedNetwork];

    //Or you wanna use the KRBPNTrainedNetwork object to recover the training data.
    KRBPNTrainedNetwork *_trainedNetwork = [[KRBPNTrainedNetwork alloc] init];
    _trainedNetwork.inputs = [NSMutableArray arrayWithObjects:
                              @[@1],
                              @[@0],
                              @[@1],
                              nil];
    [_krBPN recoverTrainedNetwork:_trainedNetwork];

    //To remove the saved trained-network.
    [_krBPN removeTrainedNetwork];

}
@end
```

## Version

V1.1

## License

MIT.
