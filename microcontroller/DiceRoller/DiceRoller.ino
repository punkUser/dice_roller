#include <Servo.h>

//---------------------------------------------------------------------------------------------

static const int k_commandNone            = 0;
static const int k_commandUp              = 1;
static const int k_commandDown            = 2;
static const int k_commandLoad            = 3;
static const int k_commandCycle           = 4;
static const int k_commandCycleDone       = 5;
static const int k_commandRangeTestUp     = 6;
static const int k_commandRangeTestDown   = 7;
static const int k_commandRangeTestValue  = 8;

static const int k_upDegrees      = 0;
static const int k_downDegrees    = 180;
static const int k_loadDegrees    = 90;

static const int k_cycleTimeMs    = 3500;

Servo g_servo;

static unsigned long g_previousTimeMs = 0;

static bool g_cycleUpNext = false;
static unsigned long g_timeSinceCycleStartMs = k_cycleTimeMs;

static int g_rangeTestMs = 1500;


//---------------------------------------------------------------------------------------------

void setup()
{
    Serial.begin(9600);
    
    g_servo.attach(9, 1000, 2000);
    g_servo.write(k_loadDegrees);

    g_previousTimeMs = millis();
}

void loop()
{
    // Detect rollover and handle somewhat gracefully
    unsigned long currentTimeMs = millis();
    unsigned long elapsedTimeMs = (currentTimeMs >= g_previousTimeMs) ? (currentTimeMs - g_previousTimeMs) : 0;
    g_previousTimeMs = currentTimeMs;

    // Cycle logic
    if (g_timeSinceCycleStartMs < k_cycleTimeMs)
    {
        g_timeSinceCycleStartMs += elapsedTimeMs;
        if (g_timeSinceCycleStartMs >= k_cycleTimeMs)
            Serial.write(k_commandCycleDone);
    }
    
    // Handle any serial input
    while (Serial.available() > 0)
    {
        // One byte = one command for simplicity for now        
        int command = Serial.read();
        if (command >= 0)
        {
            if (command == k_commandUp)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_servo.write(k_upDegrees);
            }
            else if (command == k_commandDown)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_servo.write(k_downDegrees);
            }
            else if (command == k_commandLoad)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_servo.write(k_loadDegrees);
            }
            else if (command == k_commandCycle)
            {
                // Ignore if cycle is currently still happening
                if (g_timeSinceCycleStartMs >= k_cycleTimeMs)
                {
                    if (g_cycleUpNext)
                        g_servo.write(k_upDegrees);
                    else
                        g_servo.write(k_downDegrees);
                    
                    g_timeSinceCycleStartMs = 0;
                    g_cycleUpNext = !g_cycleUpNext;
                }
            }
            else if (command == k_commandRangeTestUp)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_rangeTestMs += 10;
                g_servo.write(g_rangeTestMs);
                Serial.write(k_commandRangeTestValue);
                Serial.write(g_rangeTestMs / 10);   // 8 bit, good up to 2550ms
            }
            else if (command == k_commandRangeTestDown)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_rangeTestMs -= 10;
                g_servo.write(g_rangeTestMs);
                Serial.write(k_commandRangeTestValue);
                Serial.write(g_rangeTestMs / 10);   // 8 bit, good up to 2550ms
            }
        }
    }

    // Don't really need to be hammering the CPU/latency for this use; leave time for servo interrupts
    delay(100);
}
