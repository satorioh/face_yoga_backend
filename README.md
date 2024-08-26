# face_yoga_backend

## Overview
This project is a real-time interactive facial massage guidance system. It uses a camera to detect the user's hand gestures and actions, guiding them through a predefined facial massage routine.
<iframe src="//player.bilibili.com/player.html?aid=113027567518605&bvid=BV1XRs7e4E31&cid=500001662885653&p=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

## Core Features

### 1. Hand Position Detection
- Real-time detection of whether the user's hand is in the facial area.
- Immediate prompts if the hand is not in the facial area.
- Clear prompts, which can be text, voice, or graphical indicators.

### 2. Massage Routine Definition
- Define a standard facial massage routine, including but not limited to:
  - Cheek massage
  - Eye corner massage
  - Brow massage
  - Nose massage
  - Mouth corner massage
- Each step should have clear action definitions, including:
  - Starting position
  - Movement trajectory
  - Ending position
  - Duration or repetition count

### 3. Action Recognition and Guidance
- Real-time recognition of the user's hand movements to determine if they meet the current step's requirements.
- Immediate corrective prompts if the user's actions do not meet the requirements.
- Prompts should include text descriptions and visual guidance (e.g., animations demonstrating the correct actions).

### 4. Progress Tracking
- Track the user's completion of each massage step.
- Display current progress (e.g., completed steps/total steps).
- Automatically guide the user to the next step after completing a step.

### 5. Real-time Feedback and Preview
- Predict and display the next action based on the user's current hand position and movements.
- Overlay graphics or animations on the video feed to visually demonstrate the correct next action.

## User Interface Requirements
- Clearly display the user's real-time video feed.
- Overlay guidance information on the video feed, such as action instructions and progress bars.
- Provide clear start, pause, and end buttons.
- Consider adding settings options to allow users to customize the massage routine or adjust difficulty.

## Technical Requirements
- Use computer vision technology for accurate hand tracking and action recognition.
- Ensure real-time performance with minimal latency.
- Optimize algorithms to adapt to different lighting conditions and skin tones.
- Consider privacy protection to ensure user data security.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/satorioh/face_yoga_backend.git
   cd face_yoga_backend
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python main.py
   ```

2. Follow the on-screen instructions to start the facial massage routine.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## References
- [YouTube Video](https://www.youtube.com/watch?v=TD-_PVdRBmM)
