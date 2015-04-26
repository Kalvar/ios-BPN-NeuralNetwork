//
//  KRBPNTrainedNetwork.m
//  BPN V1.9
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014年 Kuo-Ming Lin (Kalvar). All rights reserved.
//

#import "KRBPNTrainedNetwork.h"

@interface KRBPNTrainedNetwork ()

@property (nonatomic, weak) NSCoder *_coder;

@end

@implementation KRBPNTrainedNetwork (fixNSCodings)

-(void)_encodeObject:(id)_object forKey:(NSString *)_key
{
    [self._coder encodeObject:_object forKey:_key];
}

-(void)_encodeDouble:(double)_object forKey:(NSString *)_key
{
    [self._coder encodeDouble:_object forKey:_key];
}

-(void)_encodeFloat:(float)_object forKey:(NSString *)_key
{
    [self._coder encodeFloat:_object forKey:_key];
}

-(void)_encodeInteger:(NSInteger)_object forKey:(NSString *)_key
{
    [self._coder encodeInteger:_object forKey:_key];
}

@end

@implementation KRBPNTrainedNetwork

@synthesize inputs             = _inputs;
@synthesize inputWeights       = _inputWeights;
@synthesize hiddenWeights      = _hiddenWeights;
@synthesize hiddenBiases       = _hiddenBiases;
@synthesize outputBiases       = _outputBiases;
@synthesize outputResults      = _outputResults;
@synthesize outputGoals        = _outputGoals;
@synthesize learningRate       = _learningRate;
@synthesize convergenceError   = _convergenceError;
@synthesize fOfAlpha           = _fOfAlpha;
@synthesize limitGeneration    = _limitGeneration;
@synthesize trainingGeneration = _trainingGeneration;

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRBPNTrainedNetwork *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRBPNTrainedNetwork alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

#pragma --mark NSCoding Auto Lifecycle
/*
 * @ 以下在呼叫 [self init] 時就會被自動建立
 */
-(void)encodeWithCoder:(NSCoder *)aCoder
{
    self._coder = aCoder;
    
    [self _encodeObject:_inputs forKey:@"inputs"];
    [self _encodeObject:_inputWeights forKey:@"inputWeights"];
    [self _encodeObject:_hiddenWeights forKey:@"hiddenWeights"];
    [self _encodeObject:_hiddenBiases forKey:@"hiddenBiases"];
    [self _encodeObject:_outputBiases forKey:@"outputBiases"];
    
    [self _encodeObject:_outputResults forKey:@"outputResults"];
    [self _encodeObject:_outputGoals forKey:@"outputGoals"];
    
    [self _encodeFloat:_learningRate forKey:@"learningRate"];
    [self _encodeDouble:_convergenceError forKey:@"convergenceError"];
    [self _encodeFloat:_fOfAlpha forKey:@"fOfAlpha"];
    
    [self _encodeInteger:_limitGeneration forKey:@"limitGeneration"];
    [self _encodeInteger:_trainingGeneration forKey:@"trainingGeneration"];
}

-(instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self._coder         = aDecoder;
        
        _inputs             = [aDecoder decodeObjectForKey:@"inputs"];
        _inputWeights       = [aDecoder decodeObjectForKey:@"inputWeights"];
        _hiddenWeights      = [aDecoder decodeObjectForKey:@"hiddenWeights"];
        _hiddenBiases       = [aDecoder decodeObjectForKey:@"hiddenBiases"];
        _outputBiases       = [aDecoder decodeObjectForKey:@"outputBiases"];
        
        _outputResults      = [aDecoder decodeObjectForKey:@"outputResults"];
        _outputGoals        = [aDecoder decodeObjectForKey:@"outputGoals"];
        
        _learningRate       = [aDecoder decodeFloatForKey:@"learningRate"];
        _convergenceError   = [aDecoder decodeDoubleForKey:@"convergenceError"];
        _fOfAlpha           = [aDecoder decodeFloatForKey:@"fOfAlpha"];
        
        _limitGeneration    = [aDecoder decodeIntegerForKey:@"limitGeneration"];
        _trainingGeneration = [aDecoder decodeIntegerForKey:@"trainingGeneration"];
        
    }
    return self;
}

@end
