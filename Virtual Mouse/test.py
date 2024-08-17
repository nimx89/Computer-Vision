import mouse
import keyboard
import pyautogui

x, y = mouse.get_position()
print(f"Current position: {x}, {y}")

keyboard.wait("esc")

mouse.move(x+300, y+300)
print(f"New position: {mouse.get_position()}")
keyboard.wait("esc")

print(pyautogui.size())