// Config starter code
import React from "react";
import { createChatBotMessage } from "react-chatbot-kit";

import YesNoOptions from "./widgets/options/GeneralOptions/YesNoOptions";
import ProtocolOptions from "./widgets/options/GeneralOptions/ProtocolOptions";
import ContinueOptions from "./widgets/options/GeneralOptions/ContinueOptions";
import FeelDoingOptions from "./widgets/options/GeneralOptions/FeelDoingOptions";
import EmotionOptions from "./widgets/options/GeneralOptions/EmotionOptions";

const botName = "SATbot";

const config = {
  botName: botName,
  initialMessages: [
    createChatBotMessage("Please enter your username:", {
      withAvatar: true,
      delay: 50,
    }),
  ],

  state: {
    userState: null,
    username: null,
    password: null,
    sessionID: null,
    protocols: [],
    askingForProtocol: false
  },
  customComponents: {
    header: () => <div style={{height: '16px', fontFamily: 'Arial', borderTopLeftRadius: '5px', borderTopRightRadius: '5px',
    background: '#960145', color: '#FFF5F7', padding: '8px', borderBottom: '1px solid #B8BABA'}}>CreativeBot</div>,
    botAvatar: () => <div class="react-chatbot-kit-chat-bot-avatar-container" style={{fontFamily: 'Arial'}}><p class="react-chatbot-kit-chat-bot-avatar-letter">C</p></div>
  },

  widgets: [
    {
      widgetName: "YesNo",
      widgetFunc: (props) => <YesNoOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Continue",
      widgetFunc: (props) => <ContinueOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Emotion",
      widgetFunc: (props) => <EmotionOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "FeelDoing",
      widgetFunc: (props) => <FeelDoingOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Protocol",
      widgetFunc: (props) => <ProtocolOptions {...props} />,
      mapStateToProps: ["userState", "sessionID", "protocols", "askingForProtocol"],
    },
    // {
    //   widgetName: "Feedback",
    //   widgetFunc: (props) => <FeedbackOptions {...props} />,
    //   mapStateToProps: ["userState", "sessionID"],
    // },
    // {
    //   widgetName: "YesNoProtocols",
    //   widgetFunc: (props) => <YesNoProtocolOptions {...props} />,
    //   mapStateToProps: ["userState", "sessionID"],
    // },
    // {
    //   widgetName: "RecentDistant",
    //   widgetFunc: (props) => <EventOptions {...props} />,
    //   mapStateToProps: ["userState", "sessionID"],
    // },
  ],
};

export default config;
