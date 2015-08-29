//
//  KRBPN+NSUserDefaults.m
//  BPN V2.0.4
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014年 Kuo-Ming Lin (Kalvar). All rights reserved.
//

#import "KRBPN+NSUserDefaults.h"
#import "KRBPNTrainedNetwork.h"

@implementation NSUserDefaults (Private)

+(NSUserDefaults *)userDefaults
{
    return [NSUserDefaults standardUserDefaults];
}

+(void)synchronize
{
    [[self userDefaults] synchronize];
}

@end

@implementation NSUserDefaults (ExtendUsages)

#pragma --mark Gets NSDefault Values
/*
 * @ 取出萬用型態
 */
+(id)defaultValueForKey:(NSString *)_key
{
    return [[self userDefaults] objectForKey:_key];
}

/*
 * @ 取出 String
 */
+(NSString *)stringValueForKey:(NSString *)_key
{
    return [NSString stringWithFormat:@"%@", [self defaultValueForKey:_key]];
}

/*
 * @ 取出 BOOL
 */
+(BOOL)boolValueForKey:(NSString *)_key
{
    return [[NSUserDefaults standardUserDefaults] boolForKey:_key];
}

/*
 * @ 取出 Dictionary
 */
+(NSDictionary *)dictionaryValueForKey:(NSString *)_key
{
    return [[NSUserDefaults standardUserDefaults] dictionaryForKey:_key];
}

/*
 * @ 取出 BPN Network
 */
+(KRBPNTrainedNetwork *)trainedNetworkValueForKey:(NSString *)_key
{
    NSData *_objectData = [self defaultValueForKey:_key];
    if( !_objectData )
    {
        return nil;
    }
    return (KRBPNTrainedNetwork *)[NSKeyedUnarchiver unarchiveObjectWithData:_objectData];
}

#pragma --mark Saves NSDefault Values
/*
 * @ 儲存萬用型態
 */
+(void)saveDefaultValue:(id)_value forKey:(NSString *)_forKey
{
    [[self userDefaults] setObject:_value forKey:_forKey];
    [self synchronize];
}

/*
 * @ 儲存 String
 */
+(void)saveStringValue:(NSString *)_value forKey:(NSString *)_forKey
{
    [self saveDefaultValue:_value forKey:_forKey];
}

/*
 * @ 儲存 BOOL
 */
+(void)saveBoolValue:(BOOL)_value forKey:(NSString *)_forKey
{
    [self saveDefaultValue:[NSNumber numberWithBool:_value] forKey:_forKey];
}

/*
 * @ 儲存訓練過後的 BPN Netrowk
 */
+(void)saveTrainedNetwork:(KRBPNTrainedNetwork *)_value forKey:(NSString *)_forKey
{
    if( _value )
    {
        [self saveDefaultValue:[NSKeyedArchiver archivedDataWithRootObject:_value] forKey:_forKey];
    }
}

#pragma --mark Removes NSDefault Values
+(void)removeValueForKey:(NSString *)_key
{
    [[self userDefaults] removeObjectForKey:_key];
    [self synchronize];
}

+(void)removeValuesForKeys:(NSArray *)_keys
{
    if( [_keys isKindOfClass:[NSArray class]] )
    {
        for( NSString *_key in _keys )
        {
            [[self userDefaults] removeObjectForKey:_key];
        }
        [self synchronize];
    }
}

@end