import json
import boto3
import cfnresponse

def lambda_handler(event, context):
    try:
        client = boto3.client('bedrock')
        response = client.create_knowledge_base(
            Name='MyKnowledgeBase'
        )
        knowledge_base_id = response['KnowledgeBaseId']

        # Respond to CloudFormation
        response_data = {'KnowledgeBaseID': knowledge_base_id}
        cfnresponse.send(event, context, cfnresponse.SUCCESS, response_data)
    except Exception as e:
        print(e)
        cfnresponse.send(event, context, cfnresponse.FAILED, {})
