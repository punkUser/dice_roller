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

static const int k_upServoUs      = 1130;
static const int k_loadServoUs    = 1500;
static const int k_downServoUs    = 2010;


static const int k_cycleTimeMs    = 3500;

static bool g_fullCycle           = false;   // If true, will cycle up then down each "cycle" rather than alternate

Servo g_servo;

static unsigned long g_previousTimeMs = 0;

static bool g_cycleUpNext = false;
static unsigned long g_timeSinceCycleStartMs = k_cycleTimeMs;

static int g_rangeTestUs = 1500;
static const int k_rangeTestDelta = 10;


//---------------------------------------------------------------------------------------------

void setup()
{
    Serial.begin(9600);
    
    g_servo.attach(9);
    g_servo.writeMicroseconds(k_loadServoUs);

    g_previousTimeMs = millis();
}

bool cycleNext()
{
    // Ignore if cycle is currently still happening
    if (g_timeSinceCycleStartMs >= k_cycleTimeMs)
    {
        if (g_cycleUpNext)
            g_servo.writeMicroseconds(k_upServoUs);
        else
            g_servo.writeMicroseconds(k_downServoUs);
        
        g_timeSinceCycleStartMs = 0;
        g_cycleUpNext = !g_cycleUpNext;

        return true;
    }
    else
    {
        return false;
    }
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
        {
            if (g_fullCycle)
            {
                // Only send cycle complete after a DOWN cycle
                if (g_cycleUpNext)
                    Serial.write(k_commandCycleDone);
                else
                    cycleNext();
            }
            else
            {
                // Send cycle complete after every alternation
                Serial.write(k_commandCycleDone);
            }
        }
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
                g_servo.writeMicroseconds(k_upServoUs);
            }
            else if (command == k_commandDown)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_servo.writeMicroseconds(k_downServoUs);
            }
            else if (command == k_commandLoad)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_servo.writeMicroseconds(k_loadServoUs);
            }
            else if (command == k_commandCycle)
            {
                cycleNext();
            }
            else if (command == k_commandRangeTestUp)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_rangeTestUs += k_rangeTestDelta;
                g_servo.writeMicroseconds(g_rangeTestUs);
                Serial.write(k_commandRangeTestValue);
                Serial.write(g_rangeTestUs / 10);   // 8 bit, good up to 2550ms
            }
            else if (command == k_commandRangeTestDown)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                g_rangeTestUs -= k_rangeTestDelta;
                g_servo.writeMicroseconds(g_rangeTestUs);
                Serial.write(k_commandRangeTestValue);
                Serial.write(g_rangeTestUs / 10);   // 8 bit, good up to 2550ms
            }
        }
    }

    // Don't really need to be hammering the CPU/latency for this use; leave time for servo interrupts
    delay(100);
}
