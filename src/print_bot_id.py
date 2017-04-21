import os
import json
from slackclient import SlackClient

with open('slackbot.config.json') as config:
	slackbot_config = json.load(config)

print(slackbot_config["api_token"])

BOT_NAME = "kabbot"
slack_client = SlackClient(slackbot_config["api_token"])

if __name__ == "__main__":
	api_call = slack_client.api_call("users.list")
	if api_call.get('ok'):
		users = api_call.get('members')
		for user in users:
			if 'name' in user and user.get('name') == BOT_NAME:
				print("Bot id for '" + user['name'] + "' is " + user.get('id'))
	else:
		print('could not find bot with the name ' + BOT_NAME)