import win32com.client as win32
import pythoncom


def send_email(receiver_email, subject, email_body):

    # print("Sending email to: ", receiver_email.strip())
    # print("Subject: ", subject.strip())
    # print("Email Body: ", email_body.strip())

    pythoncom.CoInitialize()

    olApp = win32.Dispatch('Outlook.Application')
    olNS = olApp.GetNameSpace('MAPI')

    # construct email item object
    mailItem = olApp.CreateItem(0)
    mailItem.Subject = subject
    mailItem.BodyFormat = 1
    mailItem.Body = email_body.strip() + "\n\n** This is an AI generated email. Please do not reply. **\n\n"
    mailItem.To = receiver_email.strip()
    mailItem.Sensitivity  = 2
    # optional (account you want to use to send the email)
    mailItem._oleobj_.Invoke(*(64209, 0, 8, 0, olNS.Accounts.Item('xxxx@yyyy.com')))
    # mailItem.Display()
    # mailItem.Save()
    mailItem.Send()
    return "Email sent successfully to: ", receiver_email.strip(), " with email body: ", email_body.strip()