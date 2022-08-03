import React from "react";
import Options from "../Options/Options";

const FeelDoingOptions = (props) => {
  const options = [
    {
      name: "Enhance creativity",
      handler: props.actionProvider.handleButtons,
      id: 5,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "FeelDoing",
    },
    {
      name: "Evaluate creativity",
      handler: props.actionProvider.handleButtons,
      id: 6,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "FeelDoing",
    },
    {
      name: "Sat protocols",
      handler: props.actionProvider.handleButtons,
      id: 7,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "FeelDoing",
    },
  ];

  return <Options options={options} />;
};
export default FeelDoingOptions;
