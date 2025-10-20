#include <Servo.h>

Servo myServo;  // Create a Servo object

void setup() {
  myServo.attach(9);  // Attach the servo to pin 9 on the Arduino
  
  // Move the motor slightly (e.g., 45 degrees)
  myServo.write(45);  
  delay(1000);        // Wait for the servo to move to 45 degrees
  
  // Now reset the motor to the neutral position (90 degrees)
  myServo.write(90);  
  delay(1000);        // Wait for the servo to reach the neutral position
}

void loop() {
  // The loop can remain empty since we are just testing and resetting the servo
}