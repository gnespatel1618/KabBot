import os
import time
import json
from slackclient import SlackClient
from kabbot import KabBot 

with open('slackbot.config.json') as config:
    slackbot_config = json.load(config)

# AT_BOT = "<@" + BOT_ID + ">"
BOT_ID = slackbot_config["bot_id"]
AT_BOT = "<@" + BOT_ID + ">"

COMMAND = "compute"

# instantiate Slack & Twilio clients
slack_client = SlackClient(slackbot_config["api_token"])
kabbot = KabBot();

def handle_command(command, channel):
	"""
	Recieves command directed at the bot and determines 
	if they are valid commands. If so then acts on the command.
	If not returns what is needed for clarification
	"""
	response = "Not sure what you mean. Use the *" + COMMAND
	# if command.startswith(COMMAND):
	response = kabbot.respond(command)
	slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)

def parse_slack_output(slack_rtm_output):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']
    return None, None

if __name__ == "__main__":
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")