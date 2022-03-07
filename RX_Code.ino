// RECIEVER CODE

#include <VirtualWire.h>
//Motor A right
const int motorPin1  = 14;  // Pin 9 of L293D
const int motorPin2  = 15;  // Pin 10 of L293D
//Motor B left
const int motorPin3  = 16; // Pin  11 of L293D
const int motorPin4  = 17;  // Pin 12 of L293D
void setup()
{
    Serial.begin(9600); 
    Serial.println("setup");
    vw_setup(2000);  
    vw_set_rx_pin(8);    
    vw_rx_start();      
    pinMode(13, OUTPUT);
    //digitalWrite(13, HIGH);
    pinMode(motorPin1, OUTPUT); 
    pinMode(motorPin2, OUTPUT);
    pinMode(motorPin3, OUTPUT);
    pinMode(motorPin4, OUTPUT);
}

void loop()
{
    uint8_t buf[VW_MAX_MESSAGE_LEN];
    uint8_t buflen = VW_MAX_MESSAGE_LEN;

    if (vw_get_message(buf, &buflen)) 
    {
  int i;
  for (i = 0; i < buflen; i++)
  {
     if(buf[i] == 'w')//if button a is pressed.... i.e.forward buton
      {
        digitalWrite(13, HIGH);
        Serial.println("Forward");
        digitalWrite(motorPin1, LOW);
        digitalWrite(motorPin2, HIGH);//FORWARD
        digitalWrite(motorPin3, LOW );
        digitalWrite(motorPin4, HIGH);
      }
     if(buf[i] == 's')
      {
        digitalWrite(13, HIGH);
        Serial.println("Backward");
        digitalWrite(motorPin1, HIGH);
        digitalWrite(motorPin2, LOW);//REVERSE
        digitalWrite(motorPin3, HIGH);
        digitalWrite(motorPin4, LOW);
      }
      if(buf[i] == 'a')
      {
        digitalWrite(13, HIGH);
        Serial.println("Left");
        digitalWrite(motorPin1, LOW);
        digitalWrite(motorPin2, LOW);//LEFT
        digitalWrite(motorPin3, LOW);
        digitalWrite(motorPin4, HIGH);
      }
     if(buf[i] == 'd')
      {
        digitalWrite(13, HIGH);
        Serial.println("Right");
        digitalWrite(motorPin1, LOW);
        digitalWrite(motorPin2, HIGH);//RIGHT
        digitalWrite(motorPin3, LOW);
        digitalWrite(motorPin4, LOW);
      }
     if(buf[i] == 'e')
      {
        digitalWrite(13, LOW);
        Serial.println("Stop");
        digitalWrite(motorPin1, LOW);
        digitalWrite(motorPin2, LOW);//STOP
        digitalWrite(motorPin3, LOW);
        digitalWrite(motorPin4, LOW);
      }
      Serial.print(" ");
  }
  Serial.println("");

   }
}

