// variables
const int NLIGHT = 12;

int LIGHT[NLIGHT];

int DELAY_YELLOW = 2;           // YELLOW LIGHT DELAY

/*
               RED LIGHT 0
                |      |
                |      |
        --------       ---------
    RED LIGHT 3         RED LIGHT 1
        --------       ---------
                |      |
                |      |
               RED LIGHT 2

                RECEIVED_DELAY_RED          RECEIVED_DELAY_GREEN     DELAY_YELLOW
    [0, 2]:     ############################-------------------------...
    [1, 3]:     -------------------------...############################
    #: Red light duration
    .: Yellow light duration
    -: Green light duration
*/

// basic functions
void setup()
{
    LIGHT[0] = 13;               // RED LIGHT 0 PIN
    LIGHT[1] = 12;               // YELLOW LIGHT 0 PIN
    LIGHT[2] = 11;               // GREEN LIGHT 0 PIN
    LIGHT[3] = 10;               // RED LIGHT 1 PIN
    LIGHT[4] = 9;                // YELLOW LIGHT 1 PIN
    LIGHT[5] = 8;                // GREEN LIGHT 1 PIN
    LIGHT[6] = 7;                // RED LIGHT 2 PIN
    LIGHT[7] = 6;                // YELLOW LIGHT 2 PIN
    LIGHT[8] = 5;                // GREEN LIGHT 2 PIN
    LIGHT[9] = 4;                // RED LIGHT 3 PIN
    LIGHT[10] = 3;               // YELLOW LIGHT 3 PIN
    LIGHT[11] = 2;               // GREEN LIGHT 3 PIN

    for (int i = 0; i < NLIGHT; i++)
    {
        pinMode(LIGHT[i], OUTPUT);
    }
    Serial.begin(9600);
    while(!Serial);
}

void loop()
{
    int received_delay_green = 0;    // green light delay received from serial
    static int whichGreen = 0;       // which green light is on, 0 = [0, 2], 1 = [1, 3]

    // RECEIVE DATA

    // wait for data
    while (Serial.available()==0){}

    // read data
    whichGreen = Serial.parseInt();

    while (Serial.available()==0){}
    received_delay_green = Serial.parseInt();

    // execute green light command
    loop_interval(received_delay_green, whichGreen);

    // Preparing for next loop
    //whichGreen = (whichGreen + 1) % 2;
    for (int i = 0; i < NLIGHT; i++)
    {
        digitalWrite(LIGHT[i], LOW);
    }
    Serial.println("Interval concluded. Waiting for new data...");
}

void set_light(int lane, int light) {
    // Set traffic light at lane <lane> to <light>
    // <light> can be 0 (red), 1 (yellow), 2 (green)
    // <lane> can be 0, 1, 2, 3 (refer to the diagram above)

    for (int i = 0; i < 3; i++) {
        int lightIndex = lane * 3 + i;
        int lightPin = LIGHT[lightIndex];
        int lightState = (light == i) ? HIGH : LOW;
        digitalWrite(lightPin, lightState);
    }    
}

void initialize_interval(int whichGreen) {
    // Initialize the interval
    //
    // <whichGreen> can be 0, 1
    // 0: Green light 0 and 2
    // 1: Green light 1 and 3
    // (refer to the diagram above)

    for (int lane = 0; lane < 4; lane++) {
        int light = (lane % 2 == whichGreen) ? 2 : 0;
        set_light(lane, light);
    }
}

void loop_interval(int duration, int whichGreen) {
    // Set corresponding lights to green for <duration> seconds, 
    // then set green lights to yellow for <DELAY_YELLOW> seconds,
    // All other lights are red.
    //
    // <whichGreen> can take value of 0 or 1
    // 0: Green light 0, Green light 2
    // 1: Green light 1, Green light 3
    // (Refer to the diagram above)
    // <duration> is the duration of the interval in seconds

    if (duration <= 0) {
        return;
    }
    initialize_interval(whichGreen);
    delay(duration * 1000);
    for (int lane = 0; lane < 4; lane++) {
        if (lane % 2 == whichGreen) {
            set_light(lane, 1);
        }
    }
    delay(DELAY_YELLOW * 1000);
}