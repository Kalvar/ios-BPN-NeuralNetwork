//
//  ViewController.m
//  BPN V1.5
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin). All rights reserved.
//

#import "ViewController.h"
#import "KRBPN.h"

@interface ViewController ()<KRBPNDelegate>

@property (nonatomic, strong) KRBPN *_krBPN;

@end

@implementation ViewController

@synthesize _krBPN;

- (void)viewDidLoad
{
    [super viewDidLoad];
    
	_krBPN          = [KRBPN sharedNetwork];
    //_krBPN.delegate = self;
    
    //各輸入向量陣列值 & 每一筆輸入向量的期望值( 輸出期望 )，因使用雙 S 曲線轉換函數，故 Input 值域須為 [0, 1]，輸出目標為 [0, 1]
    //Pattern 1
    [_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoal:0.8f];
    //Pattern 2
    [_krBPN addPatterns:@[@0, @1, @0.3, @0.9] outputGoal:0.1f];
    //Pattern 2
    [_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoal:1.0f];
    
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
    //W15, W16
    [_krBPN addPatternWeights:@[@0.2, @-0.3]];
    //W25, W26
    [_krBPN addPatternWeights:@[@0.4, @0.1]];
    //W35, W36
    [_krBPN addPatternWeights:@[@-0.5, @0.2]];
    //W45, W46
    [_krBPN addPatternWeights:@[@-0.1, @0.3]];
    
    //隱藏層神經元的偏權值 & 隱藏層神經元到輸出層神經元的權重值
    //Net 5, W57
    [_krBPN addHiddenLayerNetBias:-0.4f netWeight:-0.3f];
    //Net 6, W67
    [_krBPN addHiddenLayerNetBias:0.2f netWeight:-0.2f];
    
    //有幾顆隱藏層的神經元 ( 不用外部設定，由偏權值數目自動設定 )
    //_krBPN.countHiddens;
    //輸出層神經元偏權值, Net 6 for output
    _krBPN.outputBias       = 0.0f;
    //學習速率
    _krBPN.learningRate     = 0.8f;
    //收斂誤差值 ( 一般是 10^-3 或 10^-6 )
    _krBPN.convergenceError = 0.001f;
    //限制迭代次數
    _krBPN.limitGeneration  = 5000;
    
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
            /*
            if( !_weakKrBPN.trainedNetwork )
            {
                [_weakKrBPN saveTrainedNetwork];
            }
             */
            
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            ///*
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes)
            {
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            _weakKrBPN.inputs = [NSMutableArray arrayWithObjects:
                                 @[@0.8, @0.2, @0.2, @0.5],
                                 nil];
            [_weakKrBPN directOutput];
            //*/
        }
    }];
    
    //Remove your testing trained-network records.
    //[_krBPN removeNetwork];
    
    //Start the training, and random the weights, biases, if you use this method that you won't need to setup any weights and biases before.
    //Random means let network to auto setup inputWeights, hiddenBiases, hiddenWeights values.
    //[_krBPN trainingRandom];
    //As above said, then it will be saved the trained network after done.
    //[_krBPN trainingRandomAndSave];
    
    //Start the training network, and it won't be saving the trained-network when finished.
    [_krBPN training];
    
    //Start the training network, and it will auto-saving the trained-network when finished.
    //[_krBPN trainingSave];
    
    //If you wanna pause the training.
    //[_krBPN pause];
    
    //If you wanna continue the paused training.
    //[_krBPN continueTraining];
    
    //If you wanna reset the network back to initial situation.
    //[_krBPN reset];
    
    //When the training finished, to save the trained-network into NSUserDefaults.
    //[_krBPN saveNetwork];
    
    //If you wanna recover the trained-network data.
    //[_krBPN recoverNetwork];
    //Or you wanna use the KRBPNTrainedNetwork object to recover the training data.
    /*
    KRBPNTrainedNetwork *_trainedNetwork = [[KRBPNTrainedNetwork alloc] init];
    _trainedNetwork.inputs = [NSMutableArray arrayWithObjects:
                              @[@1],
                              @[@0],
                              @[@1],
                              nil];
    [_krBPN recoverNetwork:_trainedNetwork];
    */
    
    //To remove the saved trained-network.
    //[_krBPN removeNetwork];
}

- (void)didReceiveMemoryWarning

{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

#pragma --mark KRBPNDelegate
/*
-(void)krBPNDidTrainFinished:(KRBPN *)krBPN trainedInfo:(NSDictionary *)trainedInfo totalTimes:(NSInteger)totalTimes
{
    NSLog(@"Use trained-network to direct output : %@", krBPN.outputResults);
}

-(void)krBPNEachGeneration:(KRBPN *)krBPN trainedInfo:(NSDictionary *)trainedInfo times:(NSInteger)times
{
    NSLog(@"Generation times : %i", times);
}
 //*/


@end

