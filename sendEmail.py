import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = input("(Note: https://myaccount.google.com/lesssecureapps should be on for this email)\nType your gmail Sender email : ")  # Enter your address
password = input("Type your gmail password: ")
receiver_email = input("Type your Receiver email: ")

def sendEmail(msg,receiver_email=receiver_email):

	message = 'Subject: {}\n\n{}'.format("Facemask Notifier", msg)

	context = ssl.create_default_context()
	with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
	    server.login(sender_email, password)
	    res=server.sendmail(sender_email, receiver_email, message)

	print("mail sent to",receiver_email)