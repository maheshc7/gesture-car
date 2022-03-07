// TRANSMITTER CODE

#include <VirtualWire.h>
int cmd = 0;
void setup()
{
  Serial.begin(9600);    
  Serial.println("setup");
  vw_set_ptt_inverted(true);
  vw_set_tx_pin(13); 
  vw_setup(2000);
}
void loop()
{
  // send data only when you receive data:
  if (Serial.available() > 0) {
    // read the incoming byte:
      cmd = Serial.read();
    //cmd = 'w';
    // say what you got:
    Serial.print("I received: ");
    Serial.println(cmd, DEC);
  if(cmd=='w')
  {
   char *msg2 = "w";//send a to the receiver
   vw_send((uint8_t *)msg2, strlen(msg2));
   vw_wait_tx();
   Serial.println("w");
  }
  else if(cmd=='s')
  {
   char *msg2 = "s";
   vw_send((uint8_t *)msg2, strlen(msg2));
   vw_wait_tx();
   Serial.println("s");
  }
  else if(cmd == 'a')
  {
   char *msg2 = "a";
   vw_send((uint8_t *)msg2, strlen(msg2));
   vw_wait_tx(); 
   Serial.println("a");
  }
  else if(cmd == 'd')
  {
   char *msg2 = "d";
   vw_send((uint8_t *)msg2, strlen(msg2));
   vw_wait_tx(); 
   Serial.println("d");
  }
  else 
  {
   char *msg2 = "e";
   vw_send((uint8_t *)msg2, strlen(msg2));
   vw_wait_tx(); 
   Serial.println("e");
  }
  }
  //delay(3000);
} 
  
