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

enum Positions : int
{
    POS_LOAD = 0,
    POS_UP,
    POS_DOWN,
    POS_NUM
};
static const long g_positions[POS_NUM] = {
    0,
    (-50 * k_microstepRatio),
    ( 50 * k_microstepRatio),
};

//---------------------------------------------------------------------------------------------

static unsigned long g_previousTimeMs = 0;
static AccelStepper g_stepper(AccelStepper::DRIVER, k_stepPin, k_dirPin);

static bool g_cycleUpNext = false;
static bool g_cycleActive = false;

static long g_positionBias = 0;

//---------------------------------------------------------------------------------------------

void moveToPosition(Positions posIndex)
{
    if (g_stepper.isRunning())
        return;

    const long currentPosition = g_stepper.currentPosition();
    long newPosition = g_positions[posIndex] + g_positionBias;

    // WORKAROUND: Set/flip the direction pin manually in advance to avoid the initial step being the wrong direction
    digitalWrite(k_dirPin, (newPosition > currentPosition) ? HIGH : LOW);
    delayMicroseconds(100);
    
    g_stepper.moveTo(newPosition);
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
    if (g_cycleActive && !moving)
    {
        Serial.write(k_commandCycleDone);
        g_cycleActive = false;
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
                g_cycleActive = false;
                if (!moving)
                    moveToPosition(POS_UP);
            }
            else if (command == k_commandDown)
            {
                g_cycleActive = false;
                if (!moving)
                    moveToPosition(POS_DOWN);
            }
            else if (command == k_commandLoad)
            {
                g_cycleActive = false;
                if (!moving)
                    moveToPosition(POS_LOAD);
            }
            else if (command == k_commandCycle)
            {
                // Ignore if cycle is currently still happening
                if (!g_cycleActive && !moving)
                {
                    if (g_cycleUpNext)
                        moveToPosition(POS_UP);
                    else
                        moveToPosition(POS_DOWN);
                    
                    g_cycleActive = true;
                    g_cycleUpNext = !g_cycleUpNext;
                }
            }
            else if (command == k_commandRangeTestUp)
            {
                g_positionBias += 1;
            }
            else if (command == k_commandRangeTestDown)
            {
                g_positionBias -= 1;
            }
        }
    }
}
