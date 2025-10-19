import { Metadata } from 'next';
import { RegisterForm } from '@/components/auth/register-form';

export const metadata: Metadata = {
  title: 'Register - PathwayLens 2.0',
  description: 'Create your PathwayLens account.',
};

export default function RegisterPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <div className="w-full max-w-md space-y-8 p-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold">Create account</h1>
          <p className="text-muted-foreground mt-2">
            Get started with PathwayLens 2.0
          </p>
        </div>
        <RegisterForm />
      </div>
    </div>
  );
}
