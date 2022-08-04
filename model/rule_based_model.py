import nltk

from model.models import UserModelSession, Choice, UserModelRun, Protocol
from model.classifiers import get_emotion, get_sentence_score
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time

nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa


class ModelDecisionMaker:
    def __init__(self):

        self.dataset = pd.read_csv(
            'flowchart.csv', encoding='UTF-8')  # change path
        # self.robert = pd.read_csv('robert.csv', encoding='ISO-8859-1')
        # self.gabrielle = pd.read_csv('gabrielle.csv', encoding='ISO-8859-1')
        # self.arman = pd.read_csv('arman.csv', encoding='ISO-8859-1')
        # self.olivia = pd.read_csv('olivia.csv', encoding='ISO-8859-1')
        self.dichotomy_datasets = pd.read_csv(
            'dichotomy_exercise.csv', encoding='UTF-8')
        self.sublimation_datasets = pd.read_csv(
            'sublimate_energy_exercise.csv', encoding='UTF-8')

        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Recalling Significant Memories",
            "2: Becoming Intimate with our Child",
            "3: Singing a Song of Sffection",
            "4: Expressing Love and Care for the Child",
            "5: Pledging to Care and Support our Child",
            "6: Restoring our Emotional World after our Pledge",
            "7: Maintaining a Loving Relationship with the Child",
            "8: Creating Zest for Life",
            "9: Enjoying Nature",
            "10: Muscle Relaxation and Playful Face",
            "11: Laughing on our own",
            "12: Laughing with our Childhood self",
            "13: Creating your own Brand of Laughter",
            "14: Overcoming Current Negative Emotions",
            "15: Overcoming Past Pain",
            "16: Learning to Change our Perspective",
            "17: Learning to be playful about our past",
            "18: Identifying our Personal resentments and Acting them out",
            "19: Planning More Constructive Actions",
            "20: Updating our Beliefs to Enhance Creativity",
            "21: Praticing Affirmations",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.DICHOTOMY_TITLES = [
            "0: none",
            "1: energetic and calm",
            "2: naive and smart",
            "3: playful and disciplined",
            "4: fantasy-oriented and reality-oriented",
            "5: extroversion and introversion",
            "6: humble and proud",
            "7: masculine and feminine",
            "8: rebellious and traditionalist",
            "9: passionate and objective",
            "10: endure-pain and enjoy-life",
        ]

        self.TITLE_TO_DICHOTOMY = {
            self.DICHOTOMY_TITLES[i]: i for i in range(len(self.DICHOTOMY_TITLES))
        }

        # dichotomy exercise:
        self.DICHOTOMY_TO_EXERCISE = {
            "energetic": [
                "0: none",
                "1: cultivate an appreciation of beauty",
                "2: try to resist too much sexual activity",
                "3: wake up in the morning with a specific goal",
            ],
            "calm": [
                "0: none",
                "1: meditation",
                "2: do something relaxing",
                "3: SAT protocol 9",
            ],
            "naive": [
                "0: none",
                "1: do not assume you know everything",
                "2: stop being judgemental",
                "3: do not take appreciation so seriously",
            ],
            "smart": [
                "0: none",
                "1: time slot to read materials",
                "2: continuing education",
                "3: do brain training",
            ],
            "playful": [
                "0: none",
                "1: surprise at least one person",
                "2: experiment with your appearance",
                "3: anthropomorphizing exercise",
            ],
            "disciplined": [
                "0: none",
                "1: take charge of your schedule",
                "2: remind yourself why you want to be disciplined",
                "3: embrace the discomfort",
            ],
            "fantasy-oriented": [
                "0: none",
                "1: draw a picture of your dream world",
                "2: turn the screen off and reading the book",
                "3: loose part play",
            ],
            "reality-oriented": [
                "0: none",
                "1: delay gratification",
                "2: assess before making decision",
                "3: importance of being proactive",
            ],
            "extroversion": [
                "0: none",
                "1: SAT protocol 18",
                "2: calling instead of texting",
                "3: hang out with close friend",
            ],
            "introversion": [
                "0: none",
                "1: SAT protocol 6",
                "2: pursue a solitary hobby",
                "3: staying in on friday night",
            ],
            "humble": [
                "0: none",
                "1: practise mindfulness",
                "2: ask for help when needed",
                "3: being grateful",
            ],
            "proud": [
                "0: none",
                "1: cherished objects",
                "2: write down all the positive things",
                "3: surround yourself with positive people",
            ],
            "masculine": [
                "0: none",
                "1: voice out your opinion",
                "2: play competitive games",
                "3: embody your emotions",
            ],
            "feminine": [
                "0: none",
                "1: SAT protocol 4",
                "2: develop what you lack",
                "3: summarise a person central point",
            ],
            "rebellious": [
                "0: none",
                "1: trust your passion",
                "2: hold unpopular views",
                "3: take a chance on yourself",
            ],
            "traditionalist": [
                "0: none",
                "1: be religious for a day",
                "2: stick to the same bedtime",
                "3: be prudent on your finances",
            ],
            "passionate": [
                "0: none",
                "1: improving the complexity",
                "2: passion breeds passion",
                "3: significant motivators",
            ],
            "objective": [
                "0: none",
                "1: create a vision board",
                "2: make time for reflection",
                "3: set up a goal",
            ],
            "endure-pain": [
                "0: none",
                "1: SAT protocol 15",
                "2: SAT protocol 17",
                "3: learn to express your feelings",
            ],
            "enjoy-life": [
                "0: none",
                "1: spend money on an experience",
                "2: involve in charity",
                "3: celebrate small wins",
            ],
        }

        self.SUBLIMATION_EXERCISE = [
            "0. none",
            "1: random decide",
            "2: prioritise necessity",
            "3: experiment your peak energy period"
        ]

        self.recent_protocols = deque()
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [i for i in range(1, 22)]
        self.SAT_PROTOCOLS = [
            self.PROTOCOL_TITLES[i] for i in range(1, 22)
        ]

        # Goes from user id to actual value for dichotomy
        self.dichotomy = [i for i in range(1, 11)]
        self.dichotomy_ids = {}
        self.dichotomy_choice = []

        self.dichotomy_exercise_id = {}
        self.pole_choice = ""

        # for sublimation energy path
        self.sublimation_exercise_id = {}

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        # self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}
        self.selected_suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.remaining_choices = {}

        self.recent_questions = {}

        self.chosen_personas = {}
        self.datasets = {}

        self.QUESTIONS = {

            "ask_name": {
                "model_prompt": "Please enter your first name:",
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_name(user_id)
                },
                "protocols": {"open_text": []},
            },


            # "choose_persona": {
            #     "model_prompt": "Who would you like to talk to?",
            #     "choices": {
            #         "Kai": lambda user_id, db_session, curr_session, app: self.get_kai(user_id),
            #         "Robert": lambda user_id, db_session, curr_session, app: self.get_robert(user_id),
            #         "Gabrielle": lambda user_id, db_session, curr_session, app: self.get_gabrielle(user_id),
            #         "Arman": lambda user_id, db_session, curr_session, app: self.get_arman(user_id),
            #         "Olivia": lambda user_id, db_session, curr_session, app: self.get_olivia(user_id),
            #     },
            #     "protocols": {
            #         "Kai": [],
            #         "Robert": [],
            #         "Gabrielle": [],
            #         "Arman": [],
            #         "Olivia": [],
            #     },
            # },


            "opening_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_opening_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session
                ),
                "choices": {
                    "yes": {
                        "sad": "after_classification_negative",
                        "angry": "after_classification_negative",
                        "anxious": "after_classification_negative",
                        "happy": "after_classification_positive",
                    },
                    "no": "check_emotion",
                },
                "protocols": {
                    "yes": [],
                    "no": []
                },
            },

            "check_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(user_id, app, db_session),

                "choices": {
                    "sad": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "anxious": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "happy": lambda user_id, db_session, curr_session, app: self.get_happy_emotion(user_id),
                },
                "protocols": {
                    "sad": [],
                    "angry": [],
                    "anxious": [],
                    "happy": []
                },
            },

            # NEGATIVE EMOTIONS (SADNESS, ANGER, FEAR/ANXIETY)


            "after_classification_negative": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_believe_creative_negative(user_id, app, db_session),

                "choices": {
                    # "Yes, something happened": "event_is_recent",
                    # "No, it's just a general feeling": "more_questions",
                    "continue": "creative_domain"
                },
                "protocols": {
                    # "Yes, something happened": [],
                    # "No, it's just a general feeling": [],
                    "continue": []
                },
            },

            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_believe_creative_positive(user_id, app, db_session),

                "choices": {
                    # "Okay": "suggestions",
                    # "No, thank you": "ending_prompt"
                    "continue": "creative_domain"
                },
                "protocols": {
                    # change here?
                    # "Okay": [self.PROTOCOL_TITLES[9], self.PROTOCOL_TITLES[10], self.PROTOCOL_TITLES[11]],
                    # #[self.PROTOCOL_TITLES[k] for k in self.positive_protocols],
                    "continue": []
                },
            },

            ############################# ASKING CREATIVE DOMAIN #############################
            "creative_domain": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_creative_domain(user_id, app, db_session),

                "choices": {
                    "yes": "feel_like_doing",
                    "no": "recall_happy_memories"

                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "recall_happy_memories": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_recall_memories(user_id, app, db_session),

                "choices": {
                    "yes": "feel_like_doing",
                    "no": "suggest_domain_protocols"

                },
                "protocols": {
                    "yes": [],
                    "no": [self.PROTOCOL_TITLES[2], self.PROTOCOL_TITLES[4], self.PROTOCOL_TITLES[5], self.PROTOCOL_TITLES[6], self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[20]],
                },
            },

            ############################# ALL EMOTIONS #############################

            # "project_emotion": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(user_id, app, db_session),

            #     "choices": {
            #         "Continue": "suggestions",
            #     },
            #     "protocols": {
            #         "Continue": [],
            #     },
            # },


            "suggest_domain_protocols": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_domain_suggestions(user_id, app, db_session),

                "choices": {
                    # self.current_protocol_ids[user_id]
                    self.PROTOCOL_TITLES[k]: "trying_domain_protocol"
                    for k in self.positive_protocols
                },
                "protocols": {
                    self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                    for k in self.positive_protocols
                },
            },

            "trying_domain_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_domain_protocol(user_id, app, db_session),

                "choices": {"continue": "congratulate_on_igniting_creative_domain"},
                "protocols": {"continue": []},
            },

            "congratulate_on_igniting_creative_domain": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_congrats_ignite_creative_domain(user_id, app, db_session),

                "choices": {"continue": "reask_creative_domain"},
                "protocols": {"continue": []},
            },

            "reask_creative_domain": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_reask_creative_domain(user_id, app, db_session),

                "choices": {
                    "yes": "feel_like_doing",
                    "no": lambda user_id, db_session, curr_session,
                    app: self.determine_next_prompt_new_domain_protocol(
                        user_id, app)
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            # three paths of feel like doing

            "feel_like_doing": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_feel_like_doing(user_id, app, db_session),

                "choices": {
                    "enhance creativity": "project_childhood",
                    "evaluate creativity": "suggest_test_creativity_website",
                    "sat protocols": "suggest_sat_protocols",
                },
                "protocols": {
                    "enhance creativity": [],
                    "evaluate creativity": [],
                    "sat protocols": self.SAT_PROTOCOLS,
                },
            },

            # Evaluate creativity
            "suggest_test_creativity_website": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_test_creativity_website(user_id, app, db_session),

                "choices": {"continue": "ask_feel_creative"},
                "protocols": {"continue": []},
            },

            "ask_feel_creative": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_feel_creative(user_id, app, db_session),

                "choices": {
                    "yes": "creative_feel_better",
                    "no": "creative_feel_worse_no_change",
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "creative_feel_worse_no_change": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_creative_feel_worse_no_change(user_id, app, db_session),

                "choices": {
                    "yes": "project_childhood",
                    "no": "ending_prompt",
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "creative_feel_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_creative_feel_better(user_id, app, db_session),

                "choices": {"continue": "restart_session"},
                "protocols": {"continue": []},
            },

            "restart_session": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_restart_session(user_id, app, db_session),

                "choices": {
                    "yes": "feel_like_doing",
                    "no": "ending_prompt",
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            # SAT protocol path
            "suggest_sat_protocols": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_sat_suggestions(user_id, app, db_session),

                "choices": {
                    # self.current_protocol_ids[user_id]
                    self.PROTOCOL_TITLES[k]: "try_sat_protocols"
                    for k in self.positive_protocols
                },
                "protocols": {
                    self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                    for k in self.positive_protocols
                },
            },
            "try_sat_protocols": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_sat_protocol(user_id, app, db_session),

                "choices": {"continue": "ask_try_another_sat_protocol"},
                "protocols": {"continue": []},
            },

            "ask_try_another_sat_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_try_another_sat_protocol(user_id, app, db_session),

                "choices": {
                    "yes": lambda user_id, db_session, curr_session,
                    app: self.determine_next_prompt_new_sat_protocol(
                        user_id, app),
                    "no": "ending_prompt"
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            # Enhance creativity
            "project_childhood": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_childhood(user_id, app, db_session),

                "choices": {"continue": "ask_project_childhood_feeling"},
                "protocols": {"continue": []},
            },

            "ask_project_childhood_feeling": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_project_childhood_feeling(user_id, app, db_session),

                "choices": {
                    "yes": "three_path_creativity",
                    "no": "identify_negative_event"
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "identify_negative_event": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_identify_negative_event(user_id, app, db_session),

                "choices": {"continue": "ask_try_laugh_off"},
                "protocols": {"continue": []},
            },

            "ask_try_laugh_off": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_try_laugh_off(user_id, app, db_session),

                "choices": {
                    "yes": "ask_body_playful",
                    "no": "suggest_humorous_exercise"
                },
                "protocols": {
                    "yes": [],
                    "no": [self.PROTOCOL_TITLES[10], self.PROTOCOL_TITLES[11], self.PROTOCOL_TITLES[12], self.PROTOCOL_TITLES[13]],
                },
            },

            "suggest_humorous_exercise": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_humourous_suggestions(user_id, app, db_session),

                "choices": {
                    # self.current_protocol_ids[user_id]
                    self.PROTOCOL_TITLES[k]: "try_humourous_exercise"
                    for k in self.positive_protocols
                },
                "protocols": {
                    self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                    for k in self.positive_protocols
                },
            },

            "try_humourous_exercise": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_humourous_protocol(user_id, app, db_session),

                "choices": {"continue": "ask_body_playful"},
                "protocols": {"continue": []},
            },

            "ask_body_playful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_body_playful(user_id, app, db_session),

                "choices": {
                    "yes": "congratulate_playful",
                    "no": "ask_favourtie_song"
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "ask_favourtie_song": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_favourtie_song(user_id, app, db_session),

                "choices": {
                    "yes": "try_sing_favourite_song",
                    "no": "recommend_loving_song"
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "recommend_loving_song": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_recommend_loving_song(user_id, app, db_session),

                "choices": {"continue": "try_sing_favourite_song"},
                "protocols": {"continue": []},
            },

            "try_sing_favourite_song": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_try_sing_favourite_song(user_id, app, db_session),

                "choices": {"continue": "congratulate_playful"},
                "protocols": {"continue": []},
            },

            "congratulate_playful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_congratulate_playful(user_id, app, db_session),

                "choices": {"continue": "three_path_creativity"},
                "protocols": {"continue": []},
            },

            # Three path for creativity
            "three_path_creativity": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_three_path_creativity(user_id, app, db_session),

                "choices": {
                    "loosening deep belief": "try_sat_protocol_20",
                    "switch between dichotomy": "explain_dichotomy",
                    "sublimate energy": "why_energy_important",
                },
                "protocols": {
                    "loosening deep belief": [],
                    "switch between dichotomy": [],
                    "sublimate energy": [],
                },
            },

            # Switch between dichotomy
            "explain_dichotomy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_explain_dichotomy(user_id, app, db_session),

                "choices": {"continue": "choose_dichotomy"},
                "protocols": {"continue": []},
            },

            "choose_dichotomy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_choose_dichotomy(user_id, app, db_session),

                "choices": {
                    # self.current_protocol_ids[user_id]
                    self.DICHOTOMY_TITLES[k]: "why_dichotomy_important"
                    for k in self.dichotomy
                },
                "protocols": {
                    self.DICHOTOMY_TITLES[k]: [self.DICHOTOMY_TITLES[k]]
                    for k in self.dichotomy
                },
            },

            "why_dichotomy_important": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_why_dichotomy_important(user_id, app, db_session),

                "choices": {"continue": "ask_which_pole"},
                "protocols": {"continue": []},
            },

            "ask_which_pole": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_which_pole(user_id, app, db_session),

                "choices": {
                    "dichotomy a": lambda user_id, db_session, curr_session, app: self.get_dichotomy_a(user_id),
                    "dichotomy b": lambda user_id, db_session, curr_session, app: self.get_dichotomy_b(user_id)
                },
                "protocols": {
                    "dichotomy a": [],
                    "dichotomy b": [],
                },
            },

            "energetic": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: cultivate an appreciation of beauty": "try_dichotomy_exercise",
                    "2: try to resist too much sexual activity": "try_dichotomy_exercise",
                    "3: wake up in the morning with a specific goal": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: cultivate an appreciation of beauty": [],
                    "2: try to resist too much sexual activity": [],
                    "3: wake up in the morning with a specific goal": []
                },
            },

            "calm": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: meditation": "try_dichotomy_exercise",
                    "2: do something relaxing": "try_dichotomy_exercise",
                    "3: SAT protocol 9": "try_sat_protocol_dichotomy"
                },
                "protocols": {
                    "1: meditation": [],
                    "2: do something relaxing": [],
                    "3: SAT protocol 9": [self.PROTOCOL_TITLES[9]]
                },
            },

            "naive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: do not assume you know everything": "try_dichotomy_exercise",
                    "2: stop being judgemental": "try_dichotomy_exercise",
                    "3: do not take appreciation so seriously": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: do not assume you know everything": [],
                    "2: stop being judgemental": [],
                    "3: do not take appreciation so seriously": []
                },
            },

            "smart": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: time slot to read materials": "try_dichotomy_exercise",
                    "2: continuing education": "try_dichotomy_exercise",
                    "3: do brain training": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: time slot to read materials": [],
                    "2: continuing education": [],
                    "3: do brain training": []
                },
            },

            "playful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: surprise at least one person": "try_dichotomy_exercise",
                    "2: experiment with your appearance": "try_dichotomy_exercise",
                    "3: anthropomorphizing exercise": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: surprise at least one person": [],
                    "2: experiment with your appearance": [],
                    "3: anthropomorphizing exercise": []
                },
            },

            "disciplined": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: take charge of your schedule": "try_dichotomy_exercise",
                    "2: remind yourself why you want to be disciplined": "try_dichotomy_exercise",
                    "3: embrace the discomfort": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: take charge of your schedule": [],
                    "2: remind yourself why you want to be disciplined": [],
                    "3: embrace the discomfort": []
                },
            },

            "fantasy-oriented": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: draw a picture of your dream world": "try_dichotomy_exercise",
                    "2: turn the screen off and reading the book": "try_dichotomy_exercise",
                    "3: loose part play": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: draw a picture of your dream world": [],
                    "2: turn the screen off and reading the book": [],
                    "3: loose part play": []
                },
            },

            "reality-oriented": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: delay gratification": "try_dichotomy_exercise",
                    "2: assess before making decision": "try_dichotomy_exercise",
                    "3: importance of being proactive": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: delay gratification": [],
                    "2: assess before making decision": [],
                    "3: importance of being proactive": []
                },
            },

            "extroversion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: SAT protocol 18": "try_sat_protocol_dichotomy",
                    "2: calling instead of texting": "try_dichotomy_exercise",
                    "3: hang out with close friend": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: SAT protocol 18": [self.PROTOCOL_TITLES[18]],
                    "2: calling instead of texting": [],
                    "3: hang out with close friend": []
                },
            },

            "introversion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: SAT protocol 6": "try_sat_protocol_dichotomy",
                    "2: pursue a solitary hobby": "try_dichotomy_exercise",
                    "3: staying in on friday night": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: SAT protocol 6": [self.PROTOCOL_TITLES[6]],
                    "2: pursue a solitary hobby": [],
                    "3: staying in on friday night": []
                },
            },

            "humble": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: practise mindfulness": "try_dichotomy_exercise",
                    "2: ask for help when needed": "try_dichotomy_exercise",
                    "3: being grateful": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: practise mindfulness": [],
                    "2: ask for help when needed": [],
                    "3: being grateful": []
                },
            },

            "proud": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: cherished objects": "try_dichotomy_exercise",
                    "2: write down all the positive things": "try_dichotomy_exercise",
                    "3: surround yourself with positive people": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: cherished objects": [],
                    "2: write down all the positive things": [],
                    "3: surround yourself with positive people": []
                },
            },

            "masculine": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: voice out your opinion": "try_dichotomy_exercise",
                    "2: play competitive games": "try_dichotomy_exercise",
                    "3: embody your emotions": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: voice out your opinion": [],
                    "2: play competitive games": [],
                    "3: embody your emotions": []
                },
            },

            "feminine": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: SAT protocol 4": "try_sat_protocol_dichotomy",
                    "2: develop what you lack": "try_dichotomy_exercise",
                    "3: summarise a person central point": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: SAT protocol 4": [self.PROTOCOL_TITLES[4]],
                    "2: develop what you lack": [],
                    "3: summarise a person central point": []
                },
            },

            "rebellious": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: trust your passion": "try_dichotomy_exercise",
                    "2: hold unpopular views": "try_dichotomy_exercise",
                    "3: take a chance on yourself": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: trust your passion": [],
                    "2: hold unpopular views": [],
                    "3: take a chance on yourself": []
                },
            },

            "traditionalist": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: be religious for a day": "try_dichotomy_exercise",
                    "2: stick to the same bedtime": "try_dichotomy_exercise",
                    "3: be prudent on your finances": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: be religious for a day": [],
                    "2: stick to the same bedtime": [],
                    "3: be prudent on your finances": []
                },
            },

            "passionate": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: improving the complexity": "try_dichotomy_exercise",
                    "2: passion breeds passion": "try_dichotomy_exercise",
                    "3: significant motivators": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: improving the complexity": [],
                    "2: passion breeds passion": [],
                    "3: significant motivators": []
                },
            },

            "objective": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: create a vision board": "try_dichotomy_exercise",
                    "2: make time for reflection": "try_dichotomy_exercise",
                    "3: set up a goal": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: create a vision board": [],
                    "2: make time for reflection": [],
                    "3: set up a goal": []
                },
            },

            "endure-pain": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: SAT protocol 15": "try_sat_protocol_dichotomy",
                    "2: SAT protocol 17": "try_sat_protocol_dichotomy",
                    "3: learn to express your feelings": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: SAT protocol 15": [self.PROTOCOL_TITLES[15]],
                    "2: SAT protocol 17": [self.PROTOCOL_TITLES[17]],
                    "3: learn to express your feelings": []
                },
            },

            "enjoy-life": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_dichotomy_exercise(user_id, app, db_session),

                "choices": {
                    "1: spend money on an experience": "try_dichotomy_exercise",
                    "2: involve in charity": "try_dichotomy_exercise",
                    "3: celebrate small wins": "try_dichotomy_exercise"
                },
                "protocols": {
                    "1: spend money on an experience": [],
                    "2: involve in charity": [],
                    "3: celebrate small wins": []
                },
            },

            "try_dichotomy_exercise": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_dichotomy_exercise(user_id, app, db_session),

                "choices": {"continue": "user_found_useful"},
                "protocols": {"continue": []},
            },

            "try_sat_protocol_dichotomy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_sat_protocol_dichotomy(user_id, app, db_session),

                "choices": {"continue": "user_found_useful"},
                "protocols": {"continue": []},
            },

            # ask whether the exercise is useful in helping them feel more confident at the selected dichotomy

            "user_found_useful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "feel_better",
                    "I feel worse": "feel_worse",
                    "I feel no change": "feel_same",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },

            "feel_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Yes": lambda user_id, db_session, curr_session, app: self.determine_which_pole_previously(
                        user_id, app),
                    "No (other dichotomy route)": "choose_dichotomy",
                    "No (opposite pole)": "try_sat_protocol_16",
                    "No (end session)": "ending_prompt"
                },
                "protocols": {
                    "Yes": [],
                    "No (other dichotomy route)": [],
                    "No (opposite pole)": [],
                    "No (end session)": []
                },
            },

            "feel_worse": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),

                "choices": {
                    "Yes": lambda user_id, db_session, curr_session, app: self.determine_which_pole_previously(
                        user_id, app),
                    "No (other dichotomy route)": "choose_dichotomy",
                    "No (opposite pole)": "try_sat_protocol_16",
                    "No (end session)": "ending_prompt"
                },
                "protocols": {
                    "Yes": [],
                    "No (other dichotomy route)": [],
                    "No (opposite pole)": [],
                    "No (end session)": []
                },
            },

            "feel_same": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in this dichotomy.",
                    "That can sometimes happen but if you agree we could try another exercise and see if that is more helpful to you.",
                    "Would you like me to suggest a different exercise?"
                ],

                "choices": {
                    "Yes": lambda user_id, db_session, curr_session, app: self.determine_which_pole_previously(
                        user_id, app),
                    "No (other dichotomy route)": "choose_dichotomy",
                    "No (opposite pole)": "try_sat_protocol_16",
                    "No (end session)": "ending_prompt"
                },
                "protocols": {
                    "Yes": [],
                    "No (other dichotomy route)": [],
                    "No (opposite pole)": [],
                    "No (end session)": []
                },
            },

            "try_sat_protocol_16": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_try_sat_protocol_16(user_id, app, db_session),

                "choices": {"continue": lambda user_id, db_session, curr_session,
                            app: self.determine_next_prompt_opposite_pole(
                                user_id, app)},
                "protocols": {"continue": []},
            },

            # Loosening deep belief
            "try_sat_protocol_20": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_try_sat_protocol_20(user_id, app, db_session),

                "choices": {"continue": "ask_like_other_enhance_creativity"},
                "protocols": {"continue": []},
            },

            "ask_like_other_enhance_creativity": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_like_other_enhance_creativity(user_id, app, db_session),

                "choices": {
                    "yes": "three_path_creativity",
                    "no": "ending_prompt"
                },
                "protocols": {
                    "yes": [],
                    "no": []
                },
            },

            # Sublimate energy
            "why_energy_important": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_why_energy_important(user_id, app, db_session),

                "choices": {"continue": "suggest_sublimate_energy"},
                "protocols": {"continue": []},
            },

            "suggest_sublimate_energy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggest_sublimate_energy(user_id, app, db_session),

                "choices": {
                    "1: random decide": "try_sublimation_exercise",
                    "2: prioritise necessity": "try_sublimation_exercise",
                    "3: experiment your peak energy period": "try_sublimation_exercise"
                },
                "protocols": {
                    "1: random decide": [],
                    "2: prioritise necessity": [],
                    "3: experiment your peak energy period": []
                },
            },

            "try_sublimation_exercise": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_sublimation_exercise(user_id, app, db_session),

                "choices": {"continue": "congratulate_on_control_energy"},
                "protocols": {"continue": []},
            },

            "congratulate_on_control_energy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_congratulate_on_control_energy(user_id, app, db_session),

                "choices": {"continue": "ask_another_sublimation_exercise"},
                "protocols": {"continue": []},
            },

            "ask_another_sublimation_exercise": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ask_another_sublimation_exercise(user_id, app, db_session),

                "choices": {
                    "Yes": "suggest_sublimate_energy",
                    "No (Other path to enhance creativity)": "three_path_creativity",
                    "No (End session)": "ending_prompt"
                },
                "protocols": {
                    "Yes": [],
                    "No (Other path to enhance creativity)": [],
                    "No (End session)": []
                },
            },

            ###########################

            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id, app, db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
            },

            # "restart_prompt": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

            #     "choices": {
            #         "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
            #     },
            #     "protocols": {"open_text": []},
            # },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def clear_persona(self, user_id):
        self.chosen_personas[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    # def initialise_remaining_choices(self, user_id):
    #     self.remaining_choices[user_id] = ["displaying_antisocial_behaviour",
    #                                        "internal_persecutor_saviour", "personal_crisis", "rigid_thought"]

    def save_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
        except:  # noqa
            user_response = ""
        self.users_names[user_id] = user_response
        self.datasets[user_id] = self.dataset
        return "opening_prompt"
        # return "feel_like_doing"  # for testing purpose

    # from all the lists of protocols collected at each step of the dialogue it puts together some and returns these as suggestions
    # def get_suggestions(self, user_id, app):
    #     suggestions = []
    #     for curr_suggestions in list(self.suggestions[user_id]):
    #         if len(curr_suggestions) > 2:
    #             i, j = random.choices(range(0, len(curr_suggestions)), k=2)
    #             # weeds out some gibberish that im not sure why it's there
    #             if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES:
    #                 suggestions.extend(
    #                     [curr_suggestions[i], curr_suggestions[j]])
    #         else:
    #             suggestions.extend(curr_suggestions)
    #         suggestions = set(suggestions)
    #         suggestions = list(suggestions)
    #     # augment the suggestions if less than 4, we add random ones avoiding repetitions
    #     while len(suggestions) < 4:
    #         # we dont want to suggest protocol 6 or 11 at random here
    #         p = random.choice([i for i in range(1, 20) if i not in [6, 11]])
    #         if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
    #                 and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
    #             suggestions.append(self.PROTOCOL_TITLES[p])
    #             self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
    #     return suggestions

    def get_suggestions(self, user_id, app):
        suggestions = []
        # print(self.suggestions[user_id])
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 3:
                a, b, c = random.sample(range(0, len(curr_suggestions)), k=3)
                # weeds out some gibberish that im not sure why it's there
                if curr_suggestions[a] and curr_suggestions[b] and curr_suggestions[c] in self.PROTOCOL_TITLES:
                    suggestions.extend(
                        [curr_suggestions[a], curr_suggestions[b], curr_suggestions[c]])
            else:
                suggestions.extend(curr_suggestions)
            # suggestions = set(suggestions)
            suggestions = list(suggestions)
        # print(suggestions)

        return suggestions

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        # self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    def get_opening_prompt(self, user_id):
        # time.sleep(7)
        if self.users_names[user_id] == "":
            opening_prompt = [
                "Hello, this is CreativeBot.", "How are you feeling today?"]
        else:
            prev_qs = pd.DataFrame(
                self.recent_questions[user_id], columns=['sentences'])
            data = self.datasets[user_id]
            column = data["Hello {Bob}, this is CreativeBot. How are you feeling today?"].dropna(
            )
            my_string = self.get_best_sentence(column, prev_qs)
            if len(self.recent_questions[user_id]) < 50:
                self.recent_questions[user_id].append(my_string)
            else:
                self.recent_questions[user_id] = []
                self.recent_questions[user_id].append(my_string)
            opening_prompt = my_string.format(self.users_names[user_id])

        return self.split_sentence(opening_prompt)

    def get_restart_prompt(self, user_id):
        # time.sleep(7)
        if self.users_names[user_id] == "":
            restart_prompt = [
                "Please tell me again, how are you feeling today?"]
        else:
            restart_prompt = ["Please tell me again, " +
                              self.users_names[user_id] + ", how are you feeling today?"]
        return restart_prompt

    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)

    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["opening_prompt"]
        # (["happy", "sadness", "fear", "anger"])
        emotion = get_emotion(user_response)
        # print(emotion)
        if emotion == 'fear':
            self.guess_emotion_predictions[user_id] = 'Anxious/Scared'
            self.user_emotions[user_id] = 'Anxious'
        elif emotion == 'sadness':
            self.guess_emotion_predictions[user_id] = 'Sad'
            self.user_emotions[user_id] = 'Sad'
        elif emotion == 'anger':
            self.guess_emotion_predictions[user_id] = 'Angry'
            self.user_emotions[user_id] = 'Angry'
        else:
            self.guess_emotion_predictions[user_id] = 'Happy/Content'
            self.user_emotions[user_id] = 'Happy'
        # self.guess_emotion_predictions[user_id] = emotion
        # self.user_emotions[user_id] = emotion
        return "guess_emotion"

    def get_best_sentence(self, column, prev_qs):
        # return random.choice(column.dropna().sample(n=15).to_list()) #using random choice instead of machine learning
        maxscore = 0
        chosen = ''
        for row in column.dropna().sample(n=4):  # was 25
            fitscore = get_sentence_score(row, prev_qs)
            if fitscore > maxscore:
                maxscore = fitscore
                chosen = row
        if chosen != '':
            return chosen
        else:
            # was 25
            return random.choice(column.dropna().sample(n=4).to_list())

    # Split sentence according to ".?!"
    def split_sentence(self, sentence):
        temp_list = re.split('(?<=[.?!]) +', sentence)
        if '' in temp_list:
            temp_list.remove('')
        temp_list = [i + " " if i[-1] in [".", "?", "!"]
                     else i for i in temp_list]
        if len(temp_list) == 2:
            return [temp_list[0], temp_list[1]]
        elif len(temp_list) == 3:
            return [temp_list[0], temp_list[1], temp_list[2]]
        else:
            return [sentence]

    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["From what you have said, I feel that you are {a feeling}. Would you agree with me?"].dropna(
        )
        my_string = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(
            self.guess_emotion_predictions[user_id].lower())
        return self.split_sentence(question)

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Apologies for my mistake. Please select the emotion that best depict your current feeling from the below choices:"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Sad"
        self.user_emotions[user_id] = "Sad"
        return "after_classification_negative"

    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Angry"
        self.user_emotions[user_id] = "Angry"
        return "after_classification_negative"

    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Anxious/Scared"
        self.user_emotions[user_id] = "Anxious"
        return "after_classification_negative"

    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Happy/Content"
        self.user_emotions[user_id] = "Happy"
        return "after_classification_positive"

    # ASKING CREATIVE DOMAIN FUNCTION
    def get_model_prompt_believe_creative_positive(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Fantastic, I believe you are in the mood to hear some exciting news! Everyone can become creative. You just need to believe in yourself and pursue whatever excites you to achieve optimal creative performance."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_believe_creative_negative(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I am sorry to hear that, cheer up, I have some great news! Everyone can become creative. What you need is some faith in yourself and pursue whatever excites you to attain optimal creative performance."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ask_creative_domain(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do you currently have a creative domain in mind?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ask_recall_memories(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Please grab a happy childhood photo and recall those memories. Could you think of a creative domain you like pursuing upon recalling those memories?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # SUGGEST PROTOCOLS to REIGNITE CREATIVE DOMAIN
    def get_model_prompt_domain_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I am sorry that you could not think of any creative domain you want to pursue. Allow me to suggest the following exercises that will help you build a strong connection with your childhood and ignite a creative domain."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_trying_domain_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Please try to go through this {protocol} now, trust me, it will help you to discover a new dimension about yourself. When finished, please press 'continue'."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format(
            "SAT Protocol " + str(self.current_protocol_ids[user_id][0]))
        # print(self.recent_questions[user_id])
        return ["You have selected SAT Protocol " + str(self.current_protocol_ids[user_id][0]) + ". "] + self.split_sentence(question)

    def get_model_prompt_congrats_ignite_creative_domain(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, you are now getting closer to igniting the creative domain. Please project your feelings onto your childhood-self who pursues different domains of activity."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_reask_creative_domain(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["When you think about the feelings of your childhood, can you think about a creative domain you want to pursue?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def determine_next_prompt_new_domain_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        # if suggestions is found zero out, then refresh by adding appropriate protocols.
        if len(self.suggestions[user_id]) == 0:
            # Check if user_id already has suggestions
            try:
                self.suggestions[user_id]
            except KeyError:
                self.suggestions[user_id] = []
            protocols = [self.PROTOCOL_TITLES[2], self.PROTOCOL_TITLES[4], self.PROTOCOL_TITLES[5],
                         self.PROTOCOL_TITLES[6], self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[20]]
            if type(protocols) != list:
                self.suggestions[user_id].append(deque([protocols]))
            else:
                self.suggestions[user_id].append(deque(protocols))
        if len(self.suggestions[user_id]) > 0:
            # print(self.suggestions[user_id])
            return "suggest_domain_protocols"

    def get_model_prompt_feel_like_doing(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["What do you feel like doing now?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # Evaluate creativity
    def get_model_prompt_suggest_test_creativity_website(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["You can try to associate different words. Some of these websites offer creative games, please try one and see how creative you are."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        return self.split_sentence(question) + ["a: https://www.datcreativity.com/  ", "b: http://j.shirley.im/rat/",
                                                "c: https://www.remote-associates-test.com/", "When you are ready, please press 'continue' to move forward."]

    def get_model_prompt_ask_feel_creative(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do you feel creative after playing the game from one of the suggested websites?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_creative_feel_worse_no_change(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I am sorry you feel that way. I can suggest a few methods to help enhance your creative potential. Would you like to explore them?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_creative_feel_better(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, I feel very happy that you are feeling creative. Please do explore more in your creative domain and use that creative mind of yours to do meaningful work."].dropna()
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_restart_session(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like to restart for another session?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # Suggest Try SAT protocol
    def get_model_prompt_sat_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Let me introduce you to some useful {SAT protocols}. Try to practise these exercises as often as possible when you are free to enhance your various expressions which also helps increase your creative potential."].dropna()
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("SAT protocols")
        return self.split_sentence(question)

    def get_model_prompt_trying_sat_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Please try to go through this {protocol} now and discover an unlimited spectrum of emotions of yours. When finished, please press 'continue'."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        question = question.format(
            "SAT Protocol " + str(self.current_protocol_ids[user_id][0]))
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        return ["You have selected SAT Protocol " + str(self.current_protocol_ids[user_id][0]) + ". "] + self.split_sentence(question)

    def get_model_prompt_ask_try_another_sat_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like another {SAT protocol} to help enhance your emotions?"].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        question = question.format("SAT protocol")
        return self.split_sentence(question)

    def determine_next_prompt_new_sat_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        # if suggestions is found zero out, then refresh by adding appropriate protocols.
        if len(self.suggestions[user_id]) == 0:
            # Check if user_id already has suggestions
            try:
                self.suggestions[user_id]
            except KeyError:
                self.suggestions[user_id] = []
            protocols = self.SAT_PROTOCOLS
            if type(protocols) != list:
                self.suggestions[user_id].append(deque([protocols]))
            else:
                self.suggestions[user_id].append(deque(protocols))
        if len(self.suggestions[user_id]) > 0:
            # print(self.suggestions[user_id])
            return "suggest_sat_protocols"

    # Enhance creativity
    def get_model_prompt_project_childhood(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Creative people are playful. Please project your feelings onto your childhood self (SAT protocol 1) and start doing a play within your creative domain."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ask_project_childhood_feeling(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["When you project onto that scenario, is the first feeling that comes to your mind enjoyable?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_identify_negative_event(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["I am sorry you feel that way, I am here to heal that feeling. Please identify an event or situation that causes your negative feeling."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question) + ["When ready, press 'continue'."]

    def get_model_prompt_ask_try_laugh_off(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["It seems that this event has violated your expectations which is why you feel this way. Take a moment to take a deep breath, can you try to laugh it off?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_humourous_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["A humorous environment helps you to be playful. Please select an exercise to help you become playful."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_trying_humourous_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Please try to go through this {protocol} now and embrace any laughter that comes to your mind. When finished, please press 'continue'."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format(
            "SAT Protcol " + str(self.current_protocol_ids[user_id][0]))
        return ["You have selected SAT Protocol " + str(self.current_protocol_ids[user_id][0]) + ". "] + self.split_sentence(question)

    def get_model_prompt_ask_body_playful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Can you feel that your body is now in a playful mood?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ask_favourtie_song(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do you have a favourite song?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_recommend_loving_song(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do not worry, here are some common loving songs from my wonderful collections. Please pick your favourite one, when ready please press 'continue'."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question) + ["a: 'Love Me Like You Do' by Ellie Goulding. ", "b: 'This Magic Moment' by the Drifters. ",
                                                "c: 'Unchained Melody' by the Righteous Brothers. ", "d: 'I Will Always Love You' by Dolly Parton."]

    def get_model_prompt_try_sing_favourite_song(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Please loosen up the muscles around your mouth and eyes by moving them around and singing that song."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_congratulate_playful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, you can now enjoy being playful. Press 'continue' to move forward."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_three_path_creativity(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Here are three paths to enhance your creative potential while you maintain your playfulness. Please choose a path."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    # switching between dichotomy
    def get_model_prompt_explain_dichotomy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["As humans, we always prefer one pole of the {dichotomy} over the opposite, like loving to social rather than doing things alone. However, creative individuals are flexible between dichotomies, that is, they possess paradoxical qualities that allow them to broaden their spectrum of knowledge."].dropna()
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("dichotomy")
        return self.split_sentence(question)

    def get_model_prompt_choose_dichotomy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Which route of dichotomy would you like to explore with your playful mind?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_why_dichotomy_important(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Acquiring both characteristics from {a dichotomy} contributes to nurturing creativity. It could double your collection of responses."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        # print(self.dichotomy_ids[user_id][1])
        dichotomy_name = self.dichotomy_ids[user_id][1].split(" ")
        dichotomy_name.pop(0)
        dichotomy = " ".join(dichotomy_name)
        # remove "and" to record choices
        dichotomy_name.remove("and")
        self.dichotomy_choice = [dichotomy_name[0], dichotomy_name[1]]
        # print(self.dichotomy_choice)
        question = question.format("'" + str(dichotomy) + "'")
        return self.split_sentence(question)

    def get_model_prompt_ask_which_pole(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Which pole of this dichotomy would you like to work on?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        # print(self.dichotomy_ids[user_id][1])
        # self.dichotomy_choice = [dichotomy_name[0], dichotomy_name[1]]
        # print(self.dichotomy_choice)
        return self.split_sentence(question) + ["(Dichotomy A: {} or Dichotomy B: {})".format("'"+str(self.dichotomy_choice[0])+"'", "'"+str(self.dichotomy_choice[1])+"'")]

    def get_dichotomy_a(self, user_id):
        dichotomy_a = self.dichotomy_choice[0]
        return str(dichotomy_a)

    def get_dichotomy_b(self, user_id):
        dichotomy_b = self.dichotomy_choice[1]
        return str(dichotomy_b)

    def get_model_prompt_suggest_dichotomy_exercise(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Allow me to suggest a few exercises you could try to work on {a dichotomy}."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

        print(len(self.recent_questions[user_id]))
        question = question.format("'" + str(self.pole_choice) + "'")
        # print(self.dichotomy_ids[user_id][1])
        # self.dichotomy_choice = [dichotomy_name[0], dichotomy_name[1]]
        # print(self.dichotomy_choice)
        return self.split_sentence(question)

    def get_model_prompt_trying_dichotomy_exercise(self, user_id, app, db_session):

        # integer of choice in the list
        cur_exercise_choice = self.dichotomy_exercise_id[user_id][0]
        # current pole choice
        cur_pole_choice = self.dichotomy_exercise_id[user_id][1]
        data = self.dichotomy_datasets
        column = self.DICHOTOMY_TO_EXERCISE[cur_pole_choice][cur_exercise_choice]
        question = data[column][0]
        # question = "Please try to go through this exercise now {}. When you finish, press 'continue'".format(
        #     self.DICHOTOMY_TO_EXERCISE[cur_pole_choice][cur_exercise_choice])
        # print(
        #     self.DICHOTOMY_TO_EXERCISE[cur_pole_choice][cur_exercise_choice])
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        return self.split_sentence(question) + ["When you finish, please press 'continue'."]

    def get_model_prompt_trying_sat_protocol_dichotomy(self, user_id, app, db_session):
        # if try sat protocol then use back self.dichotomu_exercise_id
        # print(self.suggestions[user_id][0][0])
        suggested_choice = self.suggestions[user_id][0][0]
        sat_protocol = self.TITLE_TO_PROTOCOL[suggested_choice]
        # cur_pole_choice = self.dichotomy_exercise_id[user_id][1]
        question = "Please try to go through this protocol to nurture {} trait now.\
             When you finish, press 'continue'.".format("'"+str(self.pole_choice)+"'")
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        return ["You have selected SAT Protocol " + str(sat_protocol) + ". "] + self.split_sentence(question)

    def get_model_prompt_found_useful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Do you feel better or worse at being {a dichotomy} in your creative domain after having taken this exercise? I look forward to your feedback."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        # current pole choice
        cur_pole_choice = self.dichotomy_exercise_id[user_id][1]

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("'" + str(cur_pole_choice) + "'")
        return self.split_sentence(question)

    def get_model_prompt_new_better(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like another exercise to work on {a dichotomy} trait?"].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        pole_choice = self.pole_choice

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("'" + str(pole_choice) + "'")
        return ["Excellent!"] + self.split_sentence(question)

    def get_model_prompt_new_worse(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like another exercise to work on {a dichotomy} trait?"].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)
        pole_choice = self.pole_choice

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("'" + str(pole_choice) + "'")
        return ["That's Okay, don't beat yourself up."] + self.split_sentence(question)

    def determine_which_pole_previously(self, user_id, app):
        return str(self.pole_choice)

    def get_model_prompt_try_sat_protocol_16(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Please try to go through {SAT protocol 16} now to change your perspective to the other pole. When finished, please press 'continue'."].dropna(
        )
        question = self.get_best_sentence(column, prev_qs)

        # check which pole they are in before outputting the message
        if self.pole_choice == self.dichotomy_choice[0]:
            switch_pole_choice = self.dichotomy_choice[1]
        else:
            switch_pole_choice = self.dichotomy_choice[0]
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("SAT protocol 16")
        # print(self.recent_questions[user_id])
        return self.split_sentence(question) + ["(PS: other pole is {})".format("'" + str(switch_pole_choice) + "'")]

    def determine_next_prompt_opposite_pole(self, user_id, app):
        # if pole choice belong to dichotomy a, flip it around
        if self.pole_choice == self.dichotomy_choice[0]:
            self.pole_choice = self.dichotomy_choice[1]
            return self.pole_choice
        else:
            self.pole_choice = self.dichotomy_choice[0]
            return self.pole_choice

    # loosening deep belief
    def get_model_prompt_try_sat_protocol_20(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Let me introduce you to {SAT protocol 20}. This exercise will help you practise handling switching your general belief on a generic question. When finished, please press 'continue'."].dropna()
        question = self.get_best_sentence(column, prev_qs)

        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        question = question.format("SAT protocol 20")
        # print(self.recent_questions[user_id])
        return self.split_sentence(question)

    def get_model_prompt_ask_like_other_enhance_creativity(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, you are now able to articulate opposite beliefs. Would you like the other path to enhance your creative potential?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        return self.split_sentence(question)

    # sublimate energy
    def get_model_prompt_why_energy_important(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Conserving and channelling creative energy is essential for us to live up to our creative potential. Therefore, we need self-restraint to plan and channel our limited energy toward our creative domain."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        return self.split_sentence(question)

    def get_model_prompt_suggest_sublimate_energy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Allow me to me suggest certain exercises you can try to sublimate your energy realm?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        # print(self.dichotomy_ids[user_id][1])
        # self.dichotomy_choice = [dichotomy_name[0], dichotomy_name[1]]
        # print(self.dichotomy_choice)
        return self.split_sentence(question)

    def get_model_prompt_trying_sublimation_exercise(self, user_id, app, db_session):
        # integer of choice in the list
        cur_exercise_choice = self.sublimation_exercise_id[user_id][0]
        # print(cur_exercise_choice)
        data = self.sublimation_datasets
        column = self.SUBLIMATION_EXERCISE[cur_exercise_choice]
        # print(column)
        question = data[column][0]
        # question = "Please try to go through this exercise now {}. When you finish, press 'continue'".format(
        #     self.DICHOTOMY_TO_EXERCISE[cur_pole_choice][cur_exercise_choice])
        # print(
        #     self.DICHOTOMY_TO_EXERCISE[cur_pole_choice][cur_exercise_choice])
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        return self.split_sentence(question) + ["When you finish, please press 'continue'."]

    def get_model_prompt_congratulate_on_control_energy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Congratulations, you are now able to control the flow of your creative energy."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        # print(self.dichotomy_ids[user_id][1])
        # self.dichotomy_choice = [dichotomy_name[0], dichotomy_name[1]]
        # print(self.dichotomy_choice)
        return self.split_sentence(question)

    def get_model_prompt_ask_another_sublimation_exercise(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Would you like more exercises to sublimate your energy realm?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        # print(self.recent_questions[user_id])
        # print(self.dichotomy_ids[user_id][1])
        # self.dichotomy_choice = [dichotomy_name[0], dichotomy_name[1]]
        # print(self.dichotomy_choice)
        return self.split_sentence(question)

    def get_model_prompt_ending(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(
            self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Thank you for taking part. See you soon."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question) + ["You have been disconnected. Refresh the page if you would like to start over."]

    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(
                id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id

    def save_current_choice(
        self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        # restart session
        if current_choice == "ask_name":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        # removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        # Case: update suggestions for next attempt by removing relevant one
        if (
            current_choice == "suggest_domain_protocols"
            or current_choice == "suggest_sat_protocols"
            or current_choice == "suggest_humorous_exercise"
        ):

            # PRE: user_choice is a string representing a number from 1-21,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()

            # to save user current selected protocols for them to try
            self.current_protocol_ids[user_id] = [
                current_protocol, protocol_chosen.id]

            # this loop is to remove recently selected protocols from self.suggestions[user_id][0]
            curr_protocols = self.suggestions[user_id][0]
            for j in range(len(curr_protocols)):
                if curr_protocols[j] == self.PROTOCOL_TITLES[current_protocol]:
                    # print(curr_protocols[j])
                    del curr_protocols[j]
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop()
                    break
            # print(self.suggestions[user_id])

        if (
            current_choice == "choose_dichotomy"
        ):

            # PRE: user_choice is a string representing a number from 1-10,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_DICHOTOMY[user_choice]
            except KeyError:
                current_protocol = int(user_choice)
            current_protocol_name = self.DICHOTOMY_TITLES[current_protocol]

            # to save user current selected protocols for them to try
            self.dichotomy_ids[user_id] = [
                current_protocol, current_protocol_name]

        if (
            current_choice == "ask_which_pole"
        ):
            # save pole choices
            if user_choice == "dichotomy a":
                self.pole_choice = self.dichotomy_choice[0]
            else:
                self.pole_choice = self.dichotomy_choice[1]
            # print(self.pole_choice)

        if (
            current_choice == "energetic"
            or current_choice == "calm"
            or current_choice == "naive"
            or current_choice == "smart"
            or current_choice == "playful"
            or current_choice == "disciplined"
            or current_choice == "fantasy-oriented"
            or current_choice == "reality-oriented"
            or current_choice == "extroversion"
            or current_choice == "introversion"
            or current_choice == "humble"
            or current_choice == "proud"
            or current_choice == "masculine"
            or current_choice == "feminine"
            or current_choice == "rebellious"
            or current_choice == "traditionalist"
            or current_choice == "passionate"
            or current_choice == "objective"
            or current_choice == "endure-pain"
            or current_choice == "enjoy-life"
        ):

            # PRE: user_choice is a string representing a number from 1-3,
            # or the title for the corresponding protocol

            try:
                current_protocol = int(user_choice.split(":").pop(0))
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_choice + ". " + str(current_protocol),
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()

            # to save user current selected protocols for them to try
            # protocol_chosen.id is to save for later user found the protocol to be useful
            self.dichotomy_exercise_id[user_id] = [
                current_protocol, current_choice, protocol_chosen.id]
            # print(self.dichotomy_exercise_id)
            # in the case of user pikcing exercise that involves sat protocol
            # to avoid key error
            choice = self.DICHOTOMY_TO_EXERCISE[current_choice][current_protocol]
            if self.QUESTIONS[current_choice]["protocols"][choice]:
                protocol_chosen = self.QUESTIONS[current_choice]["protocols"][choice][0]
                self.clear_suggestions(user_id)
                self.update_suggestions(user_id, protocol_chosen, app)

        # # PRE: User choice is string in ["feel better", "feel worse", "feel same"]
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.dichotomy_exercise_id[user_id][2]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

        if (
            current_choice == "suggest_sublimate_energy"
        ):

            # PRE: user_choice is a string representing a number from 1-3,
            # or the title for the corresponding protocol

            try:
                current_protocol = int(user_choice.split(":").pop(0))
            except KeyError:
                current_protocol = int(user_choice)

            self.sublimation_exercise_id[user_id] = [current_protocol]

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
            # print(option_chosen)
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
        self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        # print(current_choice)
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]

        if input_type != "open_text":
            if (
                current_choice != "suggestions"
                and current_choice != "check_emotion"
                and current_choice != "after_classification_positive"
                # and current_choice != "event_is_recent"
                # and current_choice != "more_questions"
                and current_choice != "user_found_useful"
                and current_choice != "feel_better"
                and current_choice != "feel_worse"
                and current_choice != "feel_same"
                # and current_choice != "choose_persona"
                # and current_choice != "project_emotion"
                and current_choice != "after_classification_negative"
                and current_choice != "suggest_domain_protocols"
                and current_choice != "suggest_sat_protocols"
                and current_choice != "suggest_humorous_exercise"
                and current_choice != "three_path_creativity"
                and current_choice != "choose_dichotomy"
                and current_choice != "suggest_sublimate_energy"
                and current_choice != "ask_another_sublimation_exercise"
            ):
                user_choice = user_choice.lower()

            if (
                current_choice == "suggest_domain_protocols"
                or current_choice == "suggest_sat_protocols"
                or current_choice == "suggest_humorous_exercise"
            ):

                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif (
                current_choice == "choose_dichotomy"
            ):

                try:
                    current_protocol = self.TITLE_TO_DICHOTOMY[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.DICHOTOMY_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif (
                current_choice == "energetic"
                or current_choice == "calm"
                or current_choice == "naive"
                or current_choice == "smart"
                or current_choice == "playful"
                or current_choice == "disciplined"
                or current_choice == "fantasy-oriented"
                or current_choice == "reality-oriented"
                or current_choice == "extroversion"
                or current_choice == "introversion"
                or current_choice == "humble"
                or current_choice == "proud"
                or current_choice == "masculine"
                or current_choice == "feminine"
                or current_choice == "rebellious"
                or current_choice == "traditionalist"
                or current_choice == "passionate"
                or current_choice == "objective"
                or current_choice == "endure-pain"
                or current_choice == "enjoy-life"
            ):

                try:
                    current_protocol = int(user_choice.split(":").pop(0))
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.DICHOTOMY_TO_EXERCISE[current_choice][current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif (
                current_choice == "suggest_sublimate_energy"
            ):

                try:
                    current_protocol = int(user_choice.split(":").pop(0))
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.SUBLIMATION_EXERCISE[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif current_choice == "check_emotion":
                if user_choice == "sad":
                    next_choice = current_choice_for_question["sad"]
                    protocols_chosen = current_protocols["sad"]
                elif user_choice == "angry":
                    next_choice = current_choice_for_question["angry"]
                    protocols_chosen = current_protocols["angry"]
                elif user_choice == "anxious":
                    next_choice = current_choice_for_question["anxious"]
                    protocols_chosen = current_protocols["anxious"]
                else:
                    next_choice = current_choice_for_question["happy"]
                    protocols_chosen = current_protocols["happy"]
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            if current_choice == "check_emotion":
                if user_choice == "sad":
                    next_choice = current_choice_for_question["sad"]
                    protocols_chosen = current_protocols["sad"]
                elif user_choice == "angry":
                    next_choice = current_choice_for_question["angry"]
                    protocols_chosen = current_protocols["angry"]
                elif user_choice == "anxious":
                    next_choice = current_choice_for_question["anxious"]
                    protocols_chosen = current_protocols["anxious"]
                else:
                    next_choice = current_choice_for_question["happy"]
                    protocols_chosen = current_protocols["happy"]
            else:
                next_choice = current_choice_for_question["open_text"]
                protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice.lower() == "yes":
            if self.guess_emotion_predictions[user_id] == "Sad":
                next_choice = next_choice["sad"]
            elif self.guess_emotion_predictions[user_id] == "Angry":
                next_choice = next_choice["angry"]
            elif self.guess_emotion_predictions[user_id] == "Anxious/Scared":
                next_choice = next_choice["anxious"]
            else:
                next_choice = next_choice["happy"]
            # print(next_choice)

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(
                user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]

        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)

        # if current_choice is not suggestions and the length of protocols_chosen (user_selected) is larger than zero
        # this is to add protocols into self.suggestions[user_id] from eg. recall_happy_memories before suggesting protocols
        if (
            len(protocols_chosen) > 0
            and current_choice != "suggest_domain_protocols"
            and current_choice != "suggest_sat_protocols"
        ):
            # append the protocol_chosen: eg[self.PROTOCOL_TITLES[2],4,,6,7,20] into self.suggestions[user_id] in the form of a deque()
            # print(protocols_chosen)
            self.clear_suggestions(user_id)
            self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        if (
                next_choice == "suggest_domain_protocols"
                or next_choice == "suggest_sat_protocols"
                or next_choice == "suggest_humorous_exercise"
        ):

            next_choices = self.get_suggestions(user_id, app)

        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        # update current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}
