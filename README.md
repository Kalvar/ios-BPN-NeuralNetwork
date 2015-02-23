ios-BPN-Algorithm
=================

This algorithm used EBP is one kind of Back Propagation Neural Networks ( BPN ), and also it is one of neural-network algorithms.

If you need help to know how to use this network, just ask me via email. 

``` objective-c
#import "KRBPN.h"

@interface ViewController ()

@property (nonatomic, strong) KRBPN *_krBPN;

@end

@implementation ViewController

@synthesize _krBPN;

- (void)viewDidLoad
{
- (void)viewDidLoad
{
    [super viewDidLoad];
    
    _krBPN = [KRBPN sharedNetwork];
    
    /*
     * @ 各輸入向量陣列值 & 每一筆輸入向量的期望值( 輸出期望 )
     */
    //Input Pattern 1, Output Goal of Input Pattern 1
    [_krBPN addPatterns:@[@1, @2, @0.5, @1.2] outputGoal:1.0f];
    
    //Input Pattern 2
    [_krBPN addPatterns:@[@0, @1, @0.3, @-0.9] outputGoal:0.0f];
    
    //Input Pattern 3
    [_krBPN addPatterns:@[@1, @-3, @-1, @0.4] outputGoal:1.0f];

    //輸入層各向量值到隱藏層神經元的權重 ( 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組 )
    //W15, W16
    [_krBPN addPatternWeights:@[@0.2, @-0.3]];
    
    //W25, W26
    [_krBPN addPatternWeights:@[@0.4, @0.1]];
    
    //W35, W36
    [_krBPN addPatternWeights:@[@-0.5, @0.2]];
    
    //W45, W46
    [_krBPN addPatternWeights:@[@-0.1, @0.3]];
    
    /*
     * @ Setup the Hidden Layers, 設定隱藏層神經元的偏權值 & 隱藏層神經元到輸出層神經元的權重值
     */
    //Net 5, Net 6
    //第 1 層, 隱藏層神經元 Net 4 的偏權值, 隱藏層神經元 Net 4 到下一層神經元的權重值
    [_krBPN addHiddenLayerAtIndex:0 netBias:-0.4 netWeights:@[@-0.3, @0.2, @0.15]];
    [_krBPN addHiddenLayerAtIndex:0 netBias:0.2 netWeights:@[@-0.2, @0.5, @0.35]];
    //[_krBPN addHiddenLayerAtIndex:0 netBias:0.2 netWeights:@[@-0.2, @0.5, @0.2]];
    
    
    //Net 7, Net 8
    //第 2 層
    [_krBPN addHiddenLayerAtIndex:1 netBias:0.3 netWeights:@[@-0.5, @0.1]];
    [_krBPN addHiddenLayerAtIndex:1 netBias:0.7 netWeights:@[@0.2, @0.4]];
    [_krBPN addHiddenLayerAtIndex:1 netBias:0.2 netWeights:@[@-0.2, @0.5]];
    
    
    //Net 9, Net 10
    //第 3 層 (單 Output，最後的 netWeights 只需設 1 組，設多組則為多 Output Results)
    [_krBPN addHiddenLayerAtIndex:2 netBias:-0.2 netWeights:@[@0.3]];
    [_krBPN addHiddenLayerAtIndex:2 netBias:0.25 netWeights:@[@0.2]];
    
    //有幾顆隱藏層的神經元 ( 不用外部設定，由偏權值數目自動設定 )
    //_krBPN.countHiddens;
    //輸出層神經元偏權值, Net 6 for output
    _krBPN.outputBias       = 0.1f;
    //學習速率
    _krBPN.learningRate     = 0.8f;
    //收斂誤差值 ( 一般是 10^-3 或 10^-6 )
    _krBPN.convergenceError = 0.000001f;
    //限制迭代次數
    _krBPN.limitGeneration  = 500;
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    
    //每一次的迭代( Every generation-training )
    [_krBPN setEachGeneration:^(NSInteger times, NSDictionary *trainedInfo)
    {
        NSLog(@"Generation times : %i", times);
        //NSLog(@"Generation result : %f\n\n\n", [trainedInfo objectForKey:KRBPNTrainedInfoOutputResults]);
    }];
    
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes)
    {
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes)
            {
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverTrainedNetwork];
            _weakKrBPN.inputs = [NSMutableArray arrayWithObjects:
                                 @[@0, @-1, @2, @0.1],
                                 nil];
            [_weakKrBPN useTrainedNetworkToOutput];
        }
    }];
    
    //Remove your testing trained-network records.
    //[_krBPN removeTrainedNetwork];
    
    //Start the training, and random the weights, biases, if you use this method that you won't need to setup any weights and biases before.
    //Random means let network to auto setup inputWeights, hiddenBiases, hiddenWeights values.
    //[_krBPN trainingWithRandom];
    //As above said, then it will be saved the trained network after done.
    //[_krBPN trainingWithRandomAndSave];
    
    //Start the training network, and it won't be saving the trained-network when finished.
    //[_krBPN training];
    
    //Start the training network, and it will auto-saving the trained-network when finished.
    [_krBPN trainingDoneSave];
    
    //If you wanna pause the training.
    //[_krBPN pause];
    
    //If you wanna continue the paused training.
    //[_krBPN continueTraining];
    
    //If you wanna reset the network back to initial situation.
    //[_krBPN reset];
    
    //When the training finished, to save the trained-network into NSUserDefaults.
    //[_krBPN saveTrainedNetwork];
    
    //If you wanna recover the trained-network data.
    //[_krBPN recoverTrainedNetwork];
    //Or you wanna use the KRBPNTrainedNetwork object to recover the training data.
    /*
    KRBPNTrainedNetwork *_trainedNetwork = [[KRBPNTrainedNetwork alloc] init];
    [_trainedNetwork addPatterns:@[@1, @0, @0.2, @-0.5] outputGoal:1.0f];
    [_krBPN recoverTrainedNetwork:_trainedNetwork];
    */
    
    //To remove the saved trained-network.
    //[_krBPN removeTrainedNetwork];
}
@end
```

## Version

V1.4

## License

MIT.
