# Account Access

## Password reset
Use the "Forgot password" link on the login screen. A reset email is sent within 2 minutes. Reset links expire in 1 hour. If you do not receive the email, check spam and verify the email matches the one on file.

## Two-factor authentication (2FA)
We support TOTP (Authy, Google Authenticator) and SMS. Recovery codes are shown once during setup — store them safely. If a customer has lost both their device and recovery codes, identity verification by support is required (government ID + last invoice).

## Locked accounts
After 5 failed login attempts the account is temporarily locked for 15 minutes. Repeated lockouts trigger an automatic security review.

## SSO
Enterprise plans support SAML SSO with Okta, Azure AD, and Google Workspace. Setup requires an admin to configure the IdP and our SSO settings page.
