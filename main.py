import keras
import logging
import re
import sys
import random
from slackbot import settings
from slackbot.bot import Bot, respond_to, default_reply
from memory_networks import idx_word, vectorize_stories, word_idx, story_maxlen, query_maxlen, test_stories
import numpy as np


kw = {
    'format': '[%(asctime)s] %(message)s',
    'datefmt': '%m/%d/%Y %H:%M:%S',
    'level': logging.DEBUG if settings.DEBUG else logging.INFO,
    'stream': sys.stdout,
}
logging.basicConfig(**kw)
logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


description = '''
I am Nick's Keras Chat Bot.
@nick_bot_test
'''

user_story_inp = random.choice(test_stories)
model = keras.models.load_model('model.h5')
# Notes on executing Tensorflow models in a subroutine
# https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
graph = keras.backend.get_session().graph


@default_reply
def my_default_handler(message):
    message.reply(description)


@respond_to('hi', re.IGNORECASE)
def hi(message):
    message.reply('hi')


@respond_to('input (.*)', re.IGNORECASE)
def input(message, content):
    # set story
    global user_story_inp
    user_story_inp = content
    logger.info(content)

    message.reply("We updated your input")


@respond_to('qualitative', re.IGNORECASE)
def qualitative(message):
    global user_story_inp
    global graph
    with graph.as_default():
        # current_inp = random.choice(test_stories)
        current_story, current_query, current_answer = vectorize_stories([user_story_inp], word_idx, story_maxlen, query_maxlen)
        current_prediction = model.predict([current_story, current_query])
        current_prediction = idx_word[np.argmax(current_prediction)]
        message.reply(
            ' '.join([
                ' '.join(user_story_inp[0]),
                ' '.join(user_story_inp[1]),
                '| Prediction:',
                current_prediction,
                '| Ground Truth:',
                user_story_inp[2]])
        )


@respond_to('change story', re.IGNORECASE)
def shuffle_story(message):
    global user_story_inp
    user_story_inp = random.choice(test_stories)
    message.reply(' '.join(user_story_inp[0]))


@respond_to('story', re.IGNORECASE)
def shuffle_story(message):
    global user_story_inp
    message.reply(' '.join(user_story_inp[0]))


@respond_to('query (.*)', re.IGNORECASE)
def query(message, content):
    global graph
    with graph.as_default():
        logger.info(content)
        user_query_inp = content.split(' ')
        logger.info(user_query_inp)

        user_story, user_query, user_ans = vectorize_stories([[user_story_inp[0], user_query_inp, '.']], word_idx, story_maxlen, query_maxlen)
        user_prediction = model.predict([user_story, user_query])
        user_prediction = idx_word[np.argmax(user_prediction)]
        message.reply(user_prediction)


def main():
    logger.info('bot:pre_initialize')
    bot = Bot()
    logger.info('bot:post_initialize')
    logger.info('bot:pre_run')
    bot.run()
    logger.info('bot:post_run')


if __name__ == "__main__":
    main()
