// variables
const int NLIGHT = 12;

int LIGHT[NLIGHT];

int ON_AT[NLIGHT];

int RECEIVED_DELAY_GREEN = 0;    // GREEN LIGHT 1 DELAY
int DELAY_YELLOW = 2;    // YELLOW LIGHT 1 DELAY
int RECEIVED_DELAY_RED = 15;      // RED LIGHT 1 DELAY

/*
               RED LIGHT 1
                |      |
                |      |
        --------       ---------
    RED LIGHT 2         RED LIGHT 3
        --------       ---------
                |      |
                |      |
               RED LIGHT 4

                RECEIVED_DELAY_RED          RECEIVED_DELAY_GREEN     DELAY_YELLOW
    [1, 3]:     ############################-------------------------...
    [2, 4]:     -------------------------...############################
    #: Red light duration
    .: Yellow light duration
    -: Green light duration
*/

// basic functions
void setup()
{
    LIGHT[0] = 13;               // RED LIGHT 1 PIN
    LIGHT[1] = 12;               // YELLOW LIGHT 1 PIN
    LIGHT[2] = 11;               // GREEN LIGHT 1 PIN
    LIGHT[3] = 10;               // RED LIGHT 2 PIN
    LIGHT[4] = 9;                // YELLOW LIGHT 2 PIN
    LIGHT[5] = 8;                // GREEN LIGHT 2 PIN
    LIGHT[6] = 7;                // RED LIGHT 3 PIN
    LIGHT[7] = 6;                // YELLOW LIGHT 3 PIN
    LIGHT[8] = 5;                // GREEN LIGHT 3 PIN
    LIGHT[9] = 4;                // RED LIGHT 4 PIN
    LIGHT[10] = 3;               // YELLOW LIGHT 4 PIN
    LIGHT[11] = 2;               // GREEN LIGHT 4 PIN

    for (int i = 0; i < NLIGHT; i++)
    {
        pinMode(LIGHT[i], OUTPUT);
    }
    Serial.begin(9600);
    while(!Serial);
}

void loop()
{
    // reveive data
    // wait for data
    while (Serial.available()==0){}
    // read data
    RECEIVED_DELAY_GREEN = (int) Serial.parseInt();
    // check if data is valid
    while (RECEIVED_DELAY_GREEN == 0) {
        while (Serial.available()==0){}
        RECEIVED_DELAY_GREEN = (int) Serial.parseInt();
    }
    RECEIVED_DELAY_RED = 15 - RECEIVED_DELAY_GREEN;
    loop_interval();
    Serial.println("Interval concluded. Waiting for new data...");
}

void calculate_delay()
{
    for (int i = 0; i < NLIGHT; i++)
    {
        int lightCase = i % 6;
        switch (lightCase) {
            case 0:
                ON_AT[i] = 0;
                break;
            case 1:
                ON_AT[i] = RECEIVED_DELAY_RED + RECEIVED_DELAY_GREEN;
                break;
            case 2:
                ON_AT[i] = RECEIVED_DELAY_RED;
                break;
            case 3:
                ON_AT[i] = RECEIVED_DELAY_RED;
                break;
            case 4:
                ON_AT[i] = RECEIVED_DELAY_RED - DELAY_YELLOW;
                break;
            case 5:
                ON_AT[i] = 0;
                break;
        }
    }
    for (int i = 0; i < NLIGHT; i++)
    {
        Serial.println(ON_AT[i]);
    }
}

void initialize_interval() {

    digitalWrite(LIGHT[0], HIGH);
    digitalWrite(LIGHT[1], LOW);
    digitalWrite(LIGHT[2], LOW);
    digitalWrite(LIGHT[3], LOW);
    digitalWrite(LIGHT[4], LOW);
    digitalWrite(LIGHT[5], HIGH);
    digitalWrite(LIGHT[6], HIGH);
    digitalWrite(LIGHT[7], LOW);
    digitalWrite(LIGHT[8], LOW);
    digitalWrite(LIGHT[9], LOW);
    digitalWrite(LIGHT[10], LOW);
    digitalWrite(LIGHT[11], HIGH);

    calculate_delay();

    Serial.println("Interval initialized.");
}

void set_light(int lane, int light) {
    // Set traffic light at lane <lane> to <light>
    // <light> can be 0 (red), 1 (yellow), 2 (green)
    // <lane> can be 0, 1, 2, 3 (refer to the diagram above, but 0 is the top)

    for (int i = 0; i < 3; i++) {
        int lightIndex = lane * 3 + i;
        int lightPin = LIGHT[lightIndex];
        int lightState = (light == i) ? HIGH : LOW;
        digitalWrite(lightPin, lightState);
    }    
}

void loop_interval() {
    // Set interval initial state
    initialize_interval();
    int interval = RECEIVED_DELAY_RED + RECEIVED_DELAY_GREEN + DELAY_YELLOW;
    for (int secs = 0; secs < interval; secs++) {
        for (int i = 0; i < NLIGHT; i++) {
            if (secs == ON_AT[i]) {
                set_light(i / 3, i % 3);
            }
        }
        delay(1000);
    }
}