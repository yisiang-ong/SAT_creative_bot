// MessageParser starter code
class MessageParser {
  constructor(actionProvider, state) {
    this.actionProvider = actionProvider;
    this.state = state;
  }

  // This method is called inside the chatbot when it receives a message from the user.
  parse(message) {
    // Case: User has not provided id yet
    if (this.state.username == null) {
      return this.actionProvider.askForPassword(message);
    } else if (this.state.password == null) {
      return this.actionProvider.updateUserID(this.state.username, message);
    } else if (this.state.askingForProtocol && parseInt(message) >= 1 && parseInt(message) <= 21) {
        // if there is protocol, then the input_type is 1 to 21
      const choice_info = {
        user_id: this.state.userState,
        session_id: this.state.sessionID,
        user_choice: message,
        input_type: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
      };
      this.actionProvider.stopAskingForProtocol()

      return this.actionProvider.sendRequest(choice_info);
    } else if (this.state.askingForProtocol && (parseInt(message) < 1 || parseInt(message) > 21)) {
      return this.actionProvider.askForProtocol()
    } 
    // to handle if user type the choice, change input type to protocol
    else if (message === "enhance creativity"  || message === "evaluate creativity" 
    || message === "sat protocols" || message === "yes" || message === "no" || 
    message === "continue" ||  message === "loosening deep belief"
    || message === "switch between dichotomy" || message === "sublimate energy" ||
    message === "dichotomy a" || message === "dichotomy b") {
      let input_type = "Protocol";
      // console.log(input_type)
      const currentOptionToShow = this.state.currentOptionToShow
      // console.log(currentOptionToShow)
      // Case: user types when they enter text instead of selecting an option
      if ((currentOptionToShow === "Continue" && message !== "continue") ||
        (currentOptionToShow === "Emotion" && (message !== "happy" && message !== "sad" 
        && message !== "angry" && message !== "anxious")) ||
        // (currentOptionToShow === "RecentDistant" && (message !== "Recent" && message !== "Distant")) ||
        // (currentOptionToShow === "Feedback" && (message !== "Better" && message !== "Worse" && message !== "No change")) ||
        (currentOptionToShow === "Protocol" && (!this.state.protocols.includes(message))) ||
        (currentOptionToShow === "YesNo" && (message !== "yes" && message !== "no")) 
        // (currentOptionToShow === "Dichotomy" && (!this.state.dichotomy.includes(message))) ||
      ) {
        // copy last message when the user does not select an option button.
        this.actionProvider.copyLastMessage()
      } else {
        const choice_info = {
          user_id: this.state.userState,
          session_id: this.state.sessionID,
          user_choice: message,
          input_type: input_type,
        };
        return this.actionProvider.sendRequest(choice_info);
    }}
    else {
      let input_type = null;
      if (this.state.inputType.length === 1) {
        input_type = this.state.inputType[0]
      } else {
        input_type = this.state.inputType
      }
      // console.log(input_type)
      const currentOptionToShow = this.state.currentOptionToShow
      // console.log(currentOptionToShow)
      // Case: user types when they enter text instead of selecting an option
      if ((currentOptionToShow === "Continue" && message !== "continue") ||
        (currentOptionToShow === "Emotion" && (message !== "happy" && message !== "sad" && message !== "angry" && message !== "anxious")) ||
        // (currentOptionToShow === "RecentDistant" && (message !== "Recent" && message !== "Distant")) ||
        // (currentOptionToShow === "Feedback" && (message !== "Better" && message !== "Worse" && message !== "No change")) ||
        (currentOptionToShow === "Protocol" && (!this.state.protocols.includes(message))) ||
        (currentOptionToShow === "YesNo" && (message !== "yes" && message !== "no"))
        // (currentOptionToShow === "Dichotomy" && (!this.state.dichotomy.includes(message))) ||
      ) {
        // copy last message when the user does not select an option button.
        this.actionProvider.copyLastMessage()
      } else {
        const choice_info = {
          user_id: this.state.userState,
          session_id: this.state.sessionID,
          user_choice: message,
          input_type: input_type,
        };
        return this.actionProvider.sendRequest(choice_info);
      }
    }

  }
}

export default MessageParser;
