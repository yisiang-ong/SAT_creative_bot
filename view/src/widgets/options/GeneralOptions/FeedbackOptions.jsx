import React from "react";
import Options from "../Options/Options";

const FeedbackOptions = (props) => {
  const options = [
    {
      name: "Better",
      handler: props.actionProvider.handleButtons,
      id: 16,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "Feedback",
    },
    {
      name: "Worse",
      handler: props.actionProvider.handleButtons,
      id: 17,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "Feedback",
    },
    {
      name: "No Change",
      handler: props.actionProvider.handleButtons,
      id: 18,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "Feedback",
    },
  ];

  return <Options options={options} />;
};
export default FeedbackOptions;
