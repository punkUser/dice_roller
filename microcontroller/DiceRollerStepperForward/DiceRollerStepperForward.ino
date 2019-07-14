#include <Servo.h>
#include "src/AccelStepper/AccelStepper.h"

//--------------------------------------------------------------------------------------------

static const int k_commandNone            = 0;
static const int k_commandUp              = 1;
static const int k_commandDown            = 2;
static const int k_commandLoad            = 3;
static const int k_commandCycle           = 4;
static const int k_commandCycleDone       = 5;
static const int k_commandRangeTestUp     = 6;
static const int k_commandRangeTestDown   = 7;
static const int k_commandRangeTestValue  = 8;

// Stepper settings
static uint8_t k_stepPin     = 5;
static uint8_t k_dirPin      = 6;
static long k_microstepRatio = 2;

static const long k_halfCycleSteps = 50 * k_microstepRatio;

// Cycle settings
static const int k_cycleTimeMs = 3500;

//---------------------------------------------------------------------------------------------

static unsigned long g_previousTimeMs = 0;
static AccelStepper g_stepper(AccelStepper::DRIVER, k_stepPin, k_dirPin);

static unsigned long g_timeSinceCycleStartMs = k_cycleTimeMs;

static long g_nextPositionBias = 0;

//---------------------------------------------------------------------------------------------

void moveNext(bool half = false)
{
    if (g_stepper.isRunning())
        return;

    // NOTE: We only move in one direction (+ve) now so no need to fiddle with the direction
    // pin since it will just stay in one state naturally now.

    int delta = half ? k_halfCycleSteps : (k_halfCycleSteps + k_halfCycleSteps);
    delta += g_nextPositionBias;
    g_nextPositionBias = 0;

    g_stepper.move(-delta);     // Inverted direction more convenient for loading in my setup
}

void setup()
{
    Serial.begin(9600);

    g_previousTimeMs = millis();

    g_stepper.setMaxSpeed(1000);
    g_stepper.setAcceleration(350 * k_microstepRatio);
}

void loop()
{
    // Detect rollover and handle somewhat gracefully
    unsigned long currentTimeMs = millis();
    unsigned long elapsedTimeMs = (currentTimeMs >= g_previousTimeMs) ? (currentTimeMs - g_previousTimeMs) : 0;
    g_previousTimeMs = currentTimeMs;

    // Update stepper
    bool moving = g_stepper.run();

    // Cycle logic
    if (g_timeSinceCycleStartMs < k_cycleTimeMs)
    {
        g_timeSinceCycleStartMs += elapsedTimeMs;
        if (g_timeSinceCycleStartMs >= k_cycleTimeMs && !moving)
            Serial.write(k_commandCycleDone);
    }

    // Handle any serial input
    while (Serial.available() > 0)
    {
        // One byte = one command for simplicity for now        
        int command = Serial.read();
        if (command >= 0)
        {
            if (command == k_commandUp || command == k_commandDown || command == k_commandLoad)
            {
                g_timeSinceCycleStartMs = k_cycleTimeMs;
                if (!moving)
                    moveNext(command == k_commandLoad);
            }
            else if (command == k_commandCycle)
            {
                // Ignore if cycle is currently still happening
                if (g_timeSinceCycleStartMs >= k_cycleTimeMs && !moving)
                {
                    moveNext();
                    g_timeSinceCycleStartMs = 0;
                }
            }
            else if (command == k_commandRangeTestUp)
            {
                g_nextPositionBias += 1;
            }
            else if (command == k_commandRangeTestDown)
            {
                g_nextPositionBias -= 1;
            }
        }
    }
}
